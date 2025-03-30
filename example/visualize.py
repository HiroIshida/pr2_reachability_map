import matplotlib.pyplot as plt
import numpy as np
import tqdm

from pr2_reachability_map.model import Domain, load_classifier

dom = Domain()
clf = load_classifier("larm")

positive_points = []
negative_points = []
for _ in tqdm.tqdm(range(10000)):
    x = dom.sample_point()
    x[2] = 0.7
    x[3:] = 0.0
    is_reachable = clf.predict(x)
    if is_reachable:
        positive_points.append(x[:2])
    else:
        negative_points.append(x[:2])

positive_points = np.array(positive_points)
negative_points = np.array(negative_points)
fig, ax = plt.subplots()
ax.scatter(positive_points[:, 0], positive_points[:, 1], c="b")
ax.scatter(negative_points[:, 0], negative_points[:, 1], c="r")
plt.show()

# v = PyrenderViewer()
# v.add(PointCloudLink(np.array(positive_points)))
# pr2 = PR2()
# pr2.reset_manip_pose()
# pr2.torso_lift_joint.joint_angle(0.0)
# v.add(pr2)
# v.show()
# time.sleep(1000)
