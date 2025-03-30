import numpy as np
import tqdm
from skrobot.coordinates import Coordinates
from skrobot.models.pr2 import PR2

from pr2_reachability_map.model import Domain, load_classifier

clf_larm = load_classifier("larm")
pr2 = PR2(use_tight_joint_limit=False)
link_list = pr2.larm.link_list

domain = Domain()
expected_success_count = 0
expected_failure_count = 0
false_positive_count = 0
false_negative_count = 0

for _ in tqdm.tqdm(range(1000)):
    x = domain.sample_point()
    pos, rpy = x[:3], x[3:]
    target_coords = Coordinates(pos, rpy[::-1])
    res = pr2.inverse_kinematics(
        target_coords, link_list=link_list, move_target=pr2.larm_end_coords
    )
    expected_success = clf_larm.predict(x)
    success = isinstance(res, np.ndarray)
    if expected_success:
        expected_success_count += 1
        if not success:
            false_positive_count += 1
    else:
        expected_failure_count += 1
        if success:
            false_negative_count += 1

fp_rate = false_positive_count / expected_failure_count
fn_rate = false_negative_count / expected_success_count
print(f"False positive rate: {fp_rate}")
print(f"False negative rate: {fn_rate}")
