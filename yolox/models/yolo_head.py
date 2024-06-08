#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import math
from loguru import logger

import torch
import torch.nn as nn
import torch.nn.functional as F

from yolox.utils import bboxes_iou, cxcywh2xyxy, meshgrid, visualize_assign

from .losses import IOUloss
from .network_blocks import BaseConv, DWConv


class YOLOXHead(nn.Module):
    def __init__(
        self,
        num_classes,
        width=1.0,
        strides=[8, 16, 32],
        in_channels=[256, 512, 1024],
        act="silu",
        depthwise=False,
    ):
        """
        Args:
            act (str): activation type of conv. Defalut value: "silu".
            depthwise (bool): whether apply depthwise conv in conv branch. Defalut value: False.
        """
        super().__init__()

        self.num_classes = num_classes
        self.decode_in_inference = True  # for deploy, set to False

        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()
        self.stems = nn.ModuleList()
        Conv = DWConv if depthwise else BaseConv

        for i in range(len(in_channels)):
            self.stems.append(
                BaseConv(
                    in_channels=int(in_channels[i] * width),
                    out_channels=int(256 * width),
                    ksize=1,
                    stride=1,
                    act=act,
                )
            )
            self.cls_convs.append(
                nn.Sequential(
                    *[
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                    ]
                )
            )
            self.reg_convs.append(
                nn.Sequential(
                    *[
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                    ]
                )
            )
            self.cls_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=self.num_classes,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            self.reg_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=4,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            self.obj_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=1,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )

        self.use_l1 = False
        self.l1_loss = nn.L1Loss(reduction="none")
        self.bcewithlog_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.iou_loss = IOUloss(reduction="none")
        self.strides = strides
        self.grids = [torch.zeros(1)] * len(in_channels)

    def initialize_biases(self, prior_prob):
        for conv in self.cls_preds:
            b = conv.bias.view(1, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

        for conv in self.obj_preds:
            b = conv.bias.view(1, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def forward(self, xin, labels=None, imgs=None):
        outputs = []
        origin_preds = []
        x_shifts = []
        y_shifts = []
        expanded_strides = []

        for k, (cls_conv, reg_conv, stride_this_level, x) in enumerate(
            zip(self.cls_convs, self.reg_convs, self.strides, xin)
        ):
            x = self.stems[k](x)
            cls_x = x
            reg_x = x

            cls_feat = cls_conv(cls_x)
            cls_output = self.cls_preds[k](cls_feat)

            reg_feat = reg_conv(reg_x)
            reg_output = self.reg_preds[k](reg_feat)
            obj_output = self.obj_preds[k](reg_feat)

            if self.training:
                output = torch.cat([reg_output, obj_output, cls_output], 1)
                output, grid = self.get_output_and_grid(
                    output, k, stride_this_level, xin[0].type()
                )
                x_shifts.append(grid[:, :, 0])
                y_shifts.append(grid[:, :, 1])
                expanded_strides.append(
                    torch.zeros(1, grid.shape[1])
                    .fill_(stride_this_level)
                    .type_as(xin[0])
                )
                if self.use_l1:
                    batch_size = reg_output.shape[0]
                    hsize, wsize = reg_output.shape[-2:]
                    reg_output = reg_output.view(
                        batch_size, 1, 4, hsize, wsize
                    )
                    reg_output = reg_output.permute(0, 1, 3, 4, 2).reshape(
                        batch_size, -1, 4
                    )
                    origin_preds.append(reg_output.clone())

            else:
                output = torch.cat(
                    [reg_output, obj_output.sigmoid(), cls_output.sigmoid()], 1
                )

            outputs.append(output)

        if self.training:
            return self.get_losses(
                imgs,
                x_shifts,
                y_shifts,
                expanded_strides,
                labels,
                torch.cat(outputs, 1),
                origin_preds,
                dtype=xin[0].dtype,
            )
        else:
            self.hw = [x.shape[-2:] for x in outputs]
            # [batch, n_anchors_all, 85]
            outputs = torch.cat(
                [x.flatten(start_dim=2) for x in outputs], dim=2
            ).permute(0, 2, 1)
            if self.decode_in_inference:
                return self.decode_outputs(outputs, dtype=xin[0].type())
            else:
                return outputs

    def get_output_and_grid(self, output, k, stride, dtype):
        grid = self.grids[k]

        batch_size = output.shape[0]
        n_ch = 5 + self.num_classes
        hsize, wsize = output.shape[-2:]
        if grid.shape[2:4] != output.shape[2:4]:
            yv, xv = meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = torch.stack((xv, yv), 2).view(1, 1, hsize, wsize, 2).type(dtype)
            self.grids[k] = grid

        output = output.view(batch_size, 1, n_ch, hsize, wsize)
        output = output.permute(0, 1, 3, 4, 2).reshape(
            batch_size, hsize * wsize, -1
        )
        grid = grid.view(1, -1, 2)
        output[..., :2] = (output[..., :2] + grid) * stride
        output[..., 2:4] = torch.exp(output[..., 2:4]) * stride
        return output, grid

    def decode_outputs(self, outputs, dtype):
        grids = []
        strides = []
        for (hsize, wsize), stride in zip(self.hw, self.strides):
            yv, xv = meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = torch.stack((xv, yv), 2).view(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            strides.append(torch.full((*shape, 1), stride))

        grids = torch.cat(grids, dim=1).type(dtype)
        strides = torch.cat(strides, dim=1).type(dtype)

        outputs = torch.cat([
            (outputs[..., 0:2] + grids) * strides,
            torch.exp(outputs[..., 2:4]) * strides,
            outputs[..., 4:]
        ], dim=-1)
        return outputs

    def convert_xywh_to_xyxy(self, bboxes):
        """
        Convert bounding boxes from (center_x, center_y, width, height) to (x1, y1, x2, y2),
        assuming the coordinates are normalized to the range [0, 1].

        Args:
        bboxes (Tensor): A tensor containing bounding boxes of shape (N, 4),
                        where N is the number of bounding boxes and each bounding box
                        is defined as (center_x, center_y, width, height) normalized by image dimensions.

        Returns:
        Tensor: Converted bounding boxes of shape (N, 4), where each bounding box
            is defined as (x1, y1, x2, y2) also in normalized format.
        """
        x1 = bboxes[:, 0] - 0.5 * bboxes[:, 2]
        y1 = bboxes[:, 1] - 0.5 * bboxes[:, 3]
        x2 = bboxes[:, 0] + 0.5 * bboxes[:, 2]
        y2 = bboxes[:, 1] + 0.5 * bboxes[:, 3]

        return torch.stack([x1, y1, x2, y2], dim=1)

    def normalized_areas_of_contained_bboxes(self, cell_tensor, table_tensor, modifier=0, bottom_modifier=0):
        """
        Computes the normalized sum of the areas of bounding boxes that are completely contained
        within each bounding box in a tensor of multiple table bounding boxes.

        Args:
        cell_tensor (torch.Tensor): Tensor of shape (N, 4) representing bounding boxes.
        table_tensor (torch.Tensor): Tensor of shape (M, 4) representing multiple table bounding boxes.
        modifier (int): The modifier to add to each coordinate of the table bounding box.

        Returns:
        torch.Tensor: Tensor of normalized sums of the areas for each table bounding box.
        """
        normalized_areas = []

        for table_bbox in table_tensor:
            # Modify the table_bbox by adding the modifier to each coordinate
            table_bbox = table_bbox + modifier
            # Modify the y2 coordinate of the table_bbox by adding the bottom_modifier
            table_bbox[3] = table_bbox[3] + bottom_modifier
            # Check if the top-left corner of each bbox is within the current table_bbox
            top_left_inside = (cell_tensor[:, 0] >= table_bbox[0]) & (cell_tensor[:, 1] >= table_bbox[1])
            
            # Check if the bottom-right corner of each bbox is within the current table_bbox
            bottom_right_inside = (cell_tensor[:, 2] <= table_bbox[2]) & (cell_tensor[:, 3] <= table_bbox[3])
            
            # Combine both conditions to filter the tensor
            contained = top_left_inside & bottom_right_inside
            contained_bboxes = cell_tensor[contained]

            # Calculate the area of each contained bbox: (x2 - x1) * (y2 - y1)
            areas = (contained_bboxes[:, 2] - contained_bboxes[:, 0]) * (contained_bboxes[:, 3] - contained_bboxes[:, 1])

            # Sum the areas
            total_area = areas.sum()

            # Calculate the area of the current table_bbox: (x2 - x1) * (y2 - y1)
            table_area = (table_bbox[2] - table_bbox[0]) * (table_bbox[3] - table_bbox[1])

            # Normalize the total area by the area of the current table_bbox
            normalized_area = total_area / table_area if table_area != 0 else 0  # Avoid division by zero
            normalized_areas.append(normalized_area.item())

        return torch.tensor(normalized_areas)  # Convert to a tensor for output

    def check_cells_in_expanded_areas(cell_tensor, table_tensor, inner_margin=5, outer_margin=10):
        """
        Checks if the area just outside of each table bounding box contains any cells.
        
        Args:
        cell_tensor (torch.Tensor): Tensor of shape (N, 4) representing cell bounding boxes.
        table_tensor (torch.Tensor): Tensor of shape (M, 4) representing table bounding boxes.
        inner_margin (int): Inner margin for expanding the table bounding boxes.
        mu3 (int): Outer margin for further expanding the table bounding boxes.

        Returns:
        torch.Tensor: Boolean tensor indicating if the area between inner_margin and outer_margin expansions contains any cells.
        """
        results = []

        for table_bbox in table_tensor:
            # Expand table_bbox by inner_margin and outer_margin
            inner_expanded_bbox = torch.tensor([table_bbox[0] - inner_margin, table_bbox[1] - inner_margin, 
                                                table_bbox[2] + inner_margin, table_bbox[3] + inner_margin])
            outer_expanded_bbox = torch.tensor([table_bbox[0] - outer_margin, table_bbox[1] - outer_margin, 
                                                table_bbox[2] + outer_margin, table_bbox[3] + outer_margin])

            # Check for any cell in the inner_margin expanded area
            in_inner = (cell_tensor[:, 0] >= inner_expanded_bbox[0]) & (cell_tensor[:, 1] >= inner_expanded_bbox[1]) & \
                    (cell_tensor[:, 2] <= inner_expanded_bbox[2]) & (cell_tensor[:, 3] <= inner_expanded_bbox[3])

            # Check for any cell in the outer_margin expanded area
            in_outer = (cell_tensor[:, 0] >= outer_expanded_bbox[0]) & (cell_tensor[:, 1] >= outer_expanded_bbox[1]) & \
                    (cell_tensor[:, 2] <= outer_expanded_bbox[2]) & (cell_tensor[:, 3] <= outer_expanded_bbox[3])

            # Check for cells that are in the outer area but not in the inner area
            contains_cells_in_area = (in_outer & ~in_inner).any() # This will be a boolean

            results.append(contains_cells_in_area)

        return torch.tensor(results)
    

    def prep_tensors_for_constraint_loss(self, fg_masked_cls_preds, fg_masked_bbox_preds, fg_masked_obj_preds):
        """
        Prepare tensors for constraint loss by filtering bounding boxes based on confidence and converting their format.

        Args:
        fg_masked_cls_preds (torch.Tensor): Foreground masked class predictions.
        fg_masked_bbox_preds (torch.Tensor): Foreground masked bounding box predictions.
        fg_masked_obj_preds (torch.Tensor): Foreground masked objectness predictions.

        Returns:
        tuple: A tuple containing:
            - fg_masked_bbox_pred_tables_xyxy (torch.Tensor): Bounding boxes for tables converted to xyxy format.
            - fg_masked_high_conf_bbox_pred_cells_xyxy (torch.Tensor): High confidence bounding boxes for cells converted to xyxy format.
            - fg_masked_obj_preds_tables (torch.Tensor): Objectness scores for tables.
        """
        # Get the tables and cells, assuming tables are class 0 and cells class 1
        _, predicted_classes = torch.max(fg_masked_cls_preds, dim=1)
        indices_tables = (predicted_classes == 0)
        indices_cells = (predicted_classes == 1)

        fg_masked_bbox_preds_tables = fg_masked_bbox_preds[indices_tables]
        fg_masked_obj_preds_tables = fg_masked_obj_preds[indices_tables]
        fg_masked_bbox_preds_cells = fg_masked_bbox_preds[indices_cells]


        obj_conf_cells = fg_masked_obj_preds[indices_cells]
        class_conf_cells = fg_masked_cls_preds[indices_cells, 1].unsqueeze(1)  # Assuming 1 is the index for 'cells'
        
        # Only want to use high confidence cells for constraint loss
        combined_conf_cells = obj_conf_cells * class_conf_cells
        high_conf_indices = combined_conf_cells >= 0.5  # Confidence threshold
        high_conf_bbox_cells = fg_masked_bbox_preds_cells[high_conf_indices.squeeze()]

        fg_masked_bbox_pred_tables_xyxy = self.convert_xywh_to_xyxy(fg_masked_bbox_preds_tables)
        fg_masked_high_conf_bbox_pred_cells_xyxy = self.convert_xywh_to_xyxy(high_conf_bbox_cells)
        

        return fg_masked_bbox_pred_tables_xyxy, fg_masked_high_conf_bbox_pred_cells_xyxy, fg_masked_obj_preds_tables


    def constraint_loss(self, cell_tensor, table_tensor, table_objectness_tensor, mu1=5, mu2=5, mu3=10, mu4=-10, alpha=0.125, gamma=0.1):
        """
        Computes the constraint loss.

        Adapted from: Global Table Extractor (GTE): A Framework for Joint Table Identification and Cell Structure Recognition Using Visual Context
        https://arxiv.org/pdf/2005.00589v2

        Args:
        cell_tensor (torch.Tensor): Tensor of shape (N, 4) representing cell bounding boxes. x1y1x2y2 format
        table_tensor (torch.Tensor): Tensor of shape (M, 4) representing table bounding boxes. x1y1x2y2 format
        table_objectness_tensor (torch.Tensor): Tensor of shape (M, 1) representing the objectness score for each table bounding box.
        mu1 (int): The margin to add to the table bounding box (used in penalty 2)
        mu2 (int): The inner margin to select the area just outside the table bounding box (used in penalty 3)
        mu3 (int): The outer margin to select the area just outside the table bounding box (used in penalty 3)
        mu4 (int): The margin to add to the bottom of the table bounding box (used in penalty 4)
        alpha (float): The weight for the first term in the constraint loss.
        gamma (float): The weight for the second term in the constraint loss.

        Returns:
        torch.Tensor: The constraint loss.
        """
        # validate that the table_tensor and the table_objectness_tensor have the same number of rows
        if table_tensor.size(0) != table_objectness_tensor.size(0):
            raise ValueError("The table_tensor and the table_objectness_tensor must have the same number of rows.")
        # instantiate this now to enable penalty calculation later based on if statement - should probably create a sepearte function for penalty calculation
        penalty = None
        # if cell_tensor is empty, then the penalty is 1
        if cell_tensor.size(0) == 0:
            # if there are no cells, then the penalty is one
            penalty = torch.ones(table_tensor.size(0)).float()
        if table_tensor.size(0) == 0:
            # if there are no tables, then the penalty is 0
            penalty = torch.tensor(0.)
        if penalty is None:
            # if the area_of_cells_in_tables is less than alpha then the penalty is 1, otherwise 0
            area_of_cells_in_tables = self.normalized_areas_of_contained_bboxes(cell_tensor, table_tensor, modifier=0, bottom_modifier=0)
            penalty_1 = area_of_cells_in_tables < alpha
            # if the area_of_cells_just_inside_tables is less than alpha then the penalty is 1, otherwise 0
            area_of_cells_just_inside_tables = self.normalized_areas_of_contained_bboxes(cell_tensor, table_tensor, modifier=mu1, bottom_modifier=0)
            penalty_2 = area_of_cells_just_inside_tables < alpha
            # if the area just outside the table contains any cells then the penalty is 1, otherwise 0
            penalty_3 = self.check_cells_in_expanded_areas(cell_tensor, table_tensor, inner_margin=mu2, outer_margin=mu3)
            # if the area_of_cells_just_inside_bottom_tables is less than alpha then the penalty is 1, otherwise 0
            area_of_cells_just_inside_bottom_tables = self.normalized_areas_of_contained_bboxes(cell_tensor, table_tensor, modifier=0, bottom_modifier=mu4)
            penalty_4 = area_of_cells_just_inside_bottom_tables < alpha
            # perform OR operation on all the boolean penalties
            penalty = penalty_1 | penalty_2 | penalty_3 | penalty_4
            # convert penalty to float - will be 1 or 0
            penalty = penalty.float()
        # calculate loss
        constraint_loss = (penalty * table_objectness_tensor) + (gamma * (1 - penalty) * (1 - table_objectness_tensor))
        constraint_loss = constraint_loss.sum()
        return constraint_loss

    def get_losses(
        self,
        imgs,
        x_shifts,
        y_shifts,
        expanded_strides,
        labels,
        outputs,
        origin_preds,
        dtype,
    ):
        bbox_preds = outputs[:, :, :4]  # [batch, n_anchors_all, 4]
        obj_preds = outputs[:, :, 4:5]  # [batch, n_anchors_all, 1]
        cls_preds = outputs[:, :, 5:]  # [batch, n_anchors_all, n_cls]

        # calculate targets
        nlabel = (labels.sum(dim=2) > 0).sum(dim=1)  # number of objects

        total_num_anchors = outputs.shape[1]
        x_shifts = torch.cat(x_shifts, 1)  # [1, n_anchors_all]
        y_shifts = torch.cat(y_shifts, 1)  # [1, n_anchors_all]
        expanded_strides = torch.cat(expanded_strides, 1)
        if self.use_l1:
            origin_preds = torch.cat(origin_preds, 1)

        cls_targets = []
        reg_targets = []
        l1_targets = []
        obj_targets = []
        fg_masks = []

        num_fg = 0.0
        num_gts = 0.0

        for batch_idx in range(outputs.shape[0]):
            num_gt = int(nlabel[batch_idx])
            num_gts += num_gt
            if num_gt == 0:
                cls_target = outputs.new_zeros((0, self.num_classes))
                reg_target = outputs.new_zeros((0, 4))
                l1_target = outputs.new_zeros((0, 4))
                obj_target = outputs.new_zeros((total_num_anchors, 1))
                fg_mask = outputs.new_zeros(total_num_anchors).bool()
            else:
                gt_bboxes_per_image = labels[batch_idx, :num_gt, 1:5]
                gt_classes = labels[batch_idx, :num_gt, 0]
                bboxes_preds_per_image = bbox_preds[batch_idx]

                try:
                    (
                        gt_matched_classes,
                        fg_mask,
                        pred_ious_this_matching,
                        matched_gt_inds,
                        num_fg_img,
                    ) = self.get_assignments(  # noqa
                        batch_idx,
                        num_gt,
                        gt_bboxes_per_image,
                        gt_classes,
                        bboxes_preds_per_image,
                        expanded_strides,
                        x_shifts,
                        y_shifts,
                        cls_preds,
                        obj_preds,
                    )
                except RuntimeError as e:
                    # TODO: the string might change, consider a better way
                    if "CUDA out of memory. " not in str(e):
                        raise  # RuntimeError might not caused by CUDA OOM

                    logger.error(
                        "OOM RuntimeError is raised due to the huge memory cost during label assignment. \
                           CPU mode is applied in this batch. If you want to avoid this issue, \
                           try to reduce the batch size or image size."
                    )
                    torch.cuda.empty_cache()
                    (
                        gt_matched_classes,
                        fg_mask,
                        pred_ious_this_matching,
                        matched_gt_inds,
                        num_fg_img,
                    ) = self.get_assignments(  # noqa
                        batch_idx,
                        num_gt,
                        gt_bboxes_per_image,
                        gt_classes,
                        bboxes_preds_per_image,
                        expanded_strides,
                        x_shifts,
                        y_shifts,
                        cls_preds,
                        obj_preds,
                        "cpu",
                    )

                torch.cuda.empty_cache()
                num_fg += num_fg_img

                cls_target = F.one_hot(
                    gt_matched_classes.to(torch.int64), self.num_classes
                ) * pred_ious_this_matching.unsqueeze(-1)
                obj_target = fg_mask.unsqueeze(-1)
                reg_target = gt_bboxes_per_image[matched_gt_inds]
                if self.use_l1:
                    l1_target = self.get_l1_target(
                        outputs.new_zeros((num_fg_img, 4)),
                        gt_bboxes_per_image[matched_gt_inds],
                        expanded_strides[0][fg_mask],
                        x_shifts=x_shifts[0][fg_mask],
                        y_shifts=y_shifts[0][fg_mask],
                    )

            cls_targets.append(cls_target)
            reg_targets.append(reg_target)
            obj_targets.append(obj_target.to(dtype))
            fg_masks.append(fg_mask)
            if self.use_l1:
                l1_targets.append(l1_target)

        cls_targets = torch.cat(cls_targets, 0)
        reg_targets = torch.cat(reg_targets, 0)
        obj_targets = torch.cat(obj_targets, 0)
        fg_masks = torch.cat(fg_masks, 0)
        if self.use_l1:
            l1_targets = torch.cat(l1_targets, 0)

        num_fg = max(num_fg, 1)
        fg_masked_bbox_preds = bbox_preds.view(-1, 4)[fg_masks]
        fg_masked_obj_preds = obj_preds.view(-1, 1)[fg_masks]
        fg_masked_cls_preds = cls_preds.view(-1, self.num_classes)[fg_masks]
        # prep function
        fg_masked_bbox_pred_tables_xyxy, fg_masked_high_conf_bbox_pred_cells_xyxy, fg_masked_obj_preds_tables = self.prep_tensors_for_constraint_loss(fg_masked_cls_preds, fg_masked_bbox_preds, fg_masked_obj_preds)
        # calculate the constraint loss
        constraint_loss = self.constraint_loss(fg_masked_high_conf_bbox_pred_cells_xyxy, fg_masked_bbox_pred_tables_xyxy, fg_masked_obj_preds_tables)
        constraint_loss = constraint_loss / num_fg

        loss_iou = (
            self.iou_loss(fg_masked_bbox_preds, reg_targets)
        ).sum() / num_fg
        loss_obj = (
            self.bcewithlog_loss(fg_masked_obj_preds, obj_targets)
        ).sum() / num_fg
        loss_cls = (
            self.bcewithlog_loss(
                fg_masked_cls_preds, cls_targets
            )
        ).sum() / num_fg
        if self.use_l1:
            loss_l1 = (
                self.l1_loss(origin_preds.view(-1, 4)[fg_masks], l1_targets)
            ).sum() / num_fg
        else:
            loss_l1 = 0.0

        reg_weight = 5.0
        loss = reg_weight * loss_iou + loss_obj + loss_cls + loss_l1 + constraint_loss

        return (
            loss,
            reg_weight * loss_iou,
            loss_obj,
            loss_cls,
            loss_l1,
            constraint_loss,
            num_fg / max(num_gts, 1),
        )

    def get_l1_target(self, l1_target, gt, stride, x_shifts, y_shifts, eps=1e-8):
        l1_target[:, 0] = gt[:, 0] / stride - x_shifts
        l1_target[:, 1] = gt[:, 1] / stride - y_shifts
        l1_target[:, 2] = torch.log(gt[:, 2] / stride + eps)
        l1_target[:, 3] = torch.log(gt[:, 3] / stride + eps)
        return l1_target

    @torch.no_grad()
    def get_assignments(
        self,
        batch_idx,
        num_gt,
        gt_bboxes_per_image,
        gt_classes,
        bboxes_preds_per_image,
        expanded_strides,
        x_shifts,
        y_shifts,
        cls_preds,
        obj_preds,
        mode="gpu",
    ):

        if mode == "cpu":
            print("-----------Using CPU for the Current Batch-------------")
            gt_bboxes_per_image = gt_bboxes_per_image.cpu().float()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu().float()
            gt_classes = gt_classes.cpu().float()
            expanded_strides = expanded_strides.cpu().float()
            x_shifts = x_shifts.cpu()
            y_shifts = y_shifts.cpu()

        fg_mask, geometry_relation = self.get_geometry_constraint(
            gt_bboxes_per_image,
            expanded_strides,
            x_shifts,
            y_shifts,
        )

        bboxes_preds_per_image = bboxes_preds_per_image[fg_mask]
        cls_preds_ = cls_preds[batch_idx][fg_mask]
        obj_preds_ = obj_preds[batch_idx][fg_mask]
        num_in_boxes_anchor = bboxes_preds_per_image.shape[0]

        if mode == "cpu":
            gt_bboxes_per_image = gt_bboxes_per_image.cpu()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu()

        pair_wise_ious = bboxes_iou(gt_bboxes_per_image, bboxes_preds_per_image, False)

        gt_cls_per_image = (
            F.one_hot(gt_classes.to(torch.int64), self.num_classes)
            .float()
        )
        pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8)

        if mode == "cpu":
            cls_preds_, obj_preds_ = cls_preds_.cpu(), obj_preds_.cpu()

        with torch.cuda.amp.autocast(enabled=False):
            cls_preds_ = (
                cls_preds_.float().sigmoid_() * obj_preds_.float().sigmoid_()
            ).sqrt()
            pair_wise_cls_loss = F.binary_cross_entropy(
                cls_preds_.unsqueeze(0).repeat(num_gt, 1, 1),
                gt_cls_per_image.unsqueeze(1).repeat(1, num_in_boxes_anchor, 1),
                reduction="none"
            ).sum(-1)
        del cls_preds_

        cost = (
            pair_wise_cls_loss
            + 3.0 * pair_wise_ious_loss
            + float(1e6) * (~geometry_relation)
        )

        (
            num_fg,
            gt_matched_classes,
            pred_ious_this_matching,
            matched_gt_inds,
        ) = self.simota_matching(cost, pair_wise_ious, gt_classes, num_gt, fg_mask)
        del pair_wise_cls_loss, cost, pair_wise_ious, pair_wise_ious_loss

        if mode == "cpu":
            gt_matched_classes = gt_matched_classes.cuda()
            fg_mask = fg_mask.cuda()
            pred_ious_this_matching = pred_ious_this_matching.cuda()
            matched_gt_inds = matched_gt_inds.cuda()

        return (
            gt_matched_classes,
            fg_mask,
            pred_ious_this_matching,
            matched_gt_inds,
            num_fg,
        )

    def get_geometry_constraint(
        self, gt_bboxes_per_image, expanded_strides, x_shifts, y_shifts,
    ):
        """
        Calculate whether the center of an object is located in a fixed range of
        an anchor. This is used to avert inappropriate matching. It can also reduce
        the number of candidate anchors so that the GPU memory is saved.
        """
        expanded_strides_per_image = expanded_strides[0]
        x_centers_per_image = ((x_shifts[0] + 0.5) * expanded_strides_per_image).unsqueeze(0)
        y_centers_per_image = ((y_shifts[0] + 0.5) * expanded_strides_per_image).unsqueeze(0)

        # in fixed center
        center_radius = 1.5
        center_dist = expanded_strides_per_image.unsqueeze(0) * center_radius
        gt_bboxes_per_image_l = (gt_bboxes_per_image[:, 0:1]) - center_dist
        gt_bboxes_per_image_r = (gt_bboxes_per_image[:, 0:1]) + center_dist
        gt_bboxes_per_image_t = (gt_bboxes_per_image[:, 1:2]) - center_dist
        gt_bboxes_per_image_b = (gt_bboxes_per_image[:, 1:2]) + center_dist

        c_l = x_centers_per_image - gt_bboxes_per_image_l
        c_r = gt_bboxes_per_image_r - x_centers_per_image
        c_t = y_centers_per_image - gt_bboxes_per_image_t
        c_b = gt_bboxes_per_image_b - y_centers_per_image
        center_deltas = torch.stack([c_l, c_t, c_r, c_b], 2)
        is_in_centers = center_deltas.min(dim=-1).values > 0.0
        anchor_filter = is_in_centers.sum(dim=0) > 0
        geometry_relation = is_in_centers[:, anchor_filter]

        return anchor_filter, geometry_relation

    def simota_matching(self, cost, pair_wise_ious, gt_classes, num_gt, fg_mask):
        matching_matrix = torch.zeros_like(cost, dtype=torch.uint8)

        n_candidate_k = min(10, pair_wise_ious.size(1))
        topk_ious, _ = torch.topk(pair_wise_ious, n_candidate_k, dim=1)
        dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)
        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(
                cost[gt_idx], k=dynamic_ks[gt_idx], largest=False
            )
            matching_matrix[gt_idx][pos_idx] = 1

        del topk_ious, dynamic_ks, pos_idx

        anchor_matching_gt = matching_matrix.sum(0)
        # deal with the case that one anchor matches multiple ground-truths
        if anchor_matching_gt.max() > 1:
            multiple_match_mask = anchor_matching_gt > 1
            _, cost_argmin = torch.min(cost[:, multiple_match_mask], dim=0)
            matching_matrix[:, multiple_match_mask] *= 0
            matching_matrix[cost_argmin, multiple_match_mask] = 1
        fg_mask_inboxes = anchor_matching_gt > 0
        num_fg = fg_mask_inboxes.sum().item()

        fg_mask[fg_mask.clone()] = fg_mask_inboxes

        matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)
        gt_matched_classes = gt_classes[matched_gt_inds]

        pred_ious_this_matching = (matching_matrix * pair_wise_ious).sum(0)[
            fg_mask_inboxes
        ]
        return num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds

    def visualize_assign_result(self, xin, labels=None, imgs=None, save_prefix="assign_vis_"):
        # original forward logic
        outputs, x_shifts, y_shifts, expanded_strides = [], [], [], []
        # TODO: use forward logic here.

        for k, (cls_conv, reg_conv, stride_this_level, x) in enumerate(
            zip(self.cls_convs, self.reg_convs, self.strides, xin)
        ):
            x = self.stems[k](x)
            cls_x = x
            reg_x = x

            cls_feat = cls_conv(cls_x)
            cls_output = self.cls_preds[k](cls_feat)
            reg_feat = reg_conv(reg_x)
            reg_output = self.reg_preds[k](reg_feat)
            obj_output = self.obj_preds[k](reg_feat)

            output = torch.cat([reg_output, obj_output, cls_output], 1)
            output, grid = self.get_output_and_grid(output, k, stride_this_level, xin[0].type())
            x_shifts.append(grid[:, :, 0])
            y_shifts.append(grid[:, :, 1])
            expanded_strides.append(
                torch.full((1, grid.shape[1]), stride_this_level).type_as(xin[0])
            )
            outputs.append(output)

        outputs = torch.cat(outputs, 1)
        bbox_preds = outputs[:, :, :4]  # [batch, n_anchors_all, 4]
        obj_preds = outputs[:, :, 4:5]  # [batch, n_anchors_all, 1]
        cls_preds = outputs[:, :, 5:]  # [batch, n_anchors_all, n_cls]

        # calculate targets
        total_num_anchors = outputs.shape[1]
        x_shifts = torch.cat(x_shifts, 1)  # [1, n_anchors_all]
        y_shifts = torch.cat(y_shifts, 1)  # [1, n_anchors_all]
        expanded_strides = torch.cat(expanded_strides, 1)

        nlabel = (labels.sum(dim=2) > 0).sum(dim=1)  # number of objects
        for batch_idx, (img, num_gt, label) in enumerate(zip(imgs, nlabel, labels)):
            img = imgs[batch_idx].permute(1, 2, 0).to(torch.uint8)
            num_gt = int(num_gt)
            if num_gt == 0:
                fg_mask = outputs.new_zeros(total_num_anchors).bool()
            else:
                gt_bboxes_per_image = label[:num_gt, 1:5]
                gt_classes = label[:num_gt, 0]
                bboxes_preds_per_image = bbox_preds[batch_idx]
                _, fg_mask, _, matched_gt_inds, _ = self.get_assignments(  # noqa
                    batch_idx, num_gt, gt_bboxes_per_image, gt_classes,
                    bboxes_preds_per_image, expanded_strides, x_shifts,
                    y_shifts, cls_preds, obj_preds,
                )

            img = img.cpu().numpy().copy()  # copy is crucial here
            coords = torch.stack([
                ((x_shifts + 0.5) * expanded_strides).flatten()[fg_mask],
                ((y_shifts + 0.5) * expanded_strides).flatten()[fg_mask],
            ], 1)

            xyxy_boxes = cxcywh2xyxy(gt_bboxes_per_image)
            save_name = save_prefix + str(batch_idx) + ".png"
            img = visualize_assign(img, xyxy_boxes, coords, matched_gt_inds, save_name)
            logger.info(f"save img to {save_name}")
