import vidstab
import matplotlib.pyplot as plt

stabilizer = vidstab.VidStab()
stabilizer.stabilize(input_path='videos_stab/SPOILER_wiki.avi', output_path='videos_stab/stable_wiki.avi')

stabilizer.plot_trajectory()
plt.show()

stabilizer.plot_transforms()
plt.show()