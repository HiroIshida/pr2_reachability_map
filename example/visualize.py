import time

import numpy as np
import tqdm
from skrobot.model.primitives import PointCloudLink
from skrobot.models.pr2 import PR2
from skrobot.viewers import PyrenderViewer

from pr2_reachability_map.model import Domain, load_classifier

dom = Domain()
clf = load_classifier("rarm")

points = []
for _ in tqdm.tqdm(range(10000)):
    x = dom.sample_point()
    is_reachable = clf.predict(x)
    if is_reachable:
        points.append(x[:3])

v = PyrenderViewer()
v.add(PointCloudLink(np.array(points)))
pr2 = PR2()
pr2.reset_manip_pose()
pr2.torso_lift_joint.joint_angle(0.0)
v.add(pr2)
v.show()
time.sleep(1000)
