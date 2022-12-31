from gym.envs.registration import registry, register, make, spec

register(
    id='StableGrasp-v1',
    entry_point='envs.stable_grasp_env:StableGraspEnv',
    max_episode_steps=10
)

register(
    id="TactilePush-v1",
    entry_point="envs.tactile_push_env:TactilePushEnv",
    max_episode_steps=100
)

register(
    id="TactileRotation-v1",
    entry_point='envs.dclaw_rotate_env:DClawRotateEnv',
    max_episode_steps=200
)

register(
    id='Insertion-v0',
    entry_point='envs.insertion_env:InsertionEnv',
    max_episode_steps=15
)

register(
    id='Insertion-v1',
    entry_point='envs.insertion_torch_env:InsertionTorchEnv',
    max_episode_steps=15
)

register(
    id='Insertion-v2',
    entry_point='envs.insertion_full_torch_env:InsertionFullTorchEnv',
    max_episode_steps=15
)

# register(
#     id='Insertion-v3',
#     entry_point='envs.insertion_real_torch_env:InsertionRealTorchEnv',
#     max_episode_steps=15
# )

register(
    id='Insertion-v3',
    entry_point='envs.tactile_insertion_env:TactileInsertionEnv',
    max_episode_steps=15
)

# register(
#     id='Insertion-v3',
#     entry_point='envs.insertion_real_torch_grasp_force_env_render:InsertionRealTorchEnv',
#     max_episode_steps=15
# )

# register(
#     id='Insertion-v3',
#     entry_point='envs.insertion_real_torch_rotation_only_env:InsertionRealTorchEnv',
#     max_episode_steps=15
# )