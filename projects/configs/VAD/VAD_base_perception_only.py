_base_ = ['./vad_perception_dataset.py']

model = dict(
    type='VADPerception',
    pts_bbox_head=dict(
        type='VADPerceptionHead',
        ego_his_encoder=None,
        ego_agent_decoder=None,
        ego_map_decoder=None,
        motion_decoder=None,
        motion_map_decoder=None,
        loss_traj=dict(type='L1Loss', loss_weight=0.0),
        loss_traj_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=0.0),
        loss_plan_reg=dict(type='L1Loss', loss_weight=0.0),
        loss_plan_bound=dict(type='PlanMapBoundLoss', loss_weight=0.0),
        loss_plan_col=dict(type='PlanCollisionLoss', loss_weight=0.0),
        loss_plan_dir=dict(type='PlanMapDirectionLoss', loss_weight=0.0)
    )
)
