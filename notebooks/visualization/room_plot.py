#%%
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Computer Modern'
# Room dimensions
room_length = 7  # in meters
room_width = 6  # in meters

# Loudspeaker positions (x, y)
loudspeakers = np.linspace(-180,150,12)/180*np.pi

# Microphone positions (x, y)
dist = 0.15
lsdist = 2
FS = 14
LW = 1.5
microphones_x = [-2.5, -2.5, 2.5, 2.5, -1.5, 1.5]
microphones_y = [0.5, -0.5, 0.5, -0.5, 0, 0]

colors = sns.color_palette("colorblind").as_hex()


color_m = colors[0]
color_n = colors[3]
color_h = colors[1]
color_s = colors[2]

# color_m = '#4e79a7' # Dark Blue
# color_n = '#f28e2b' # Vermillion Orange
# color_h = '#76b7b2' # Teal
# color_s = '#59a14f' # Olive Green
# color_n = '#d62728' # Scarlet Red
# color_h = '#bcbd22' # sage green
# color_s = '#9467bd' # Purple

# Plot the room
fig, ax = plt.subplots(dpi=1200)
ax.set_xlim(0, room_length)
ax.set_ylim(0, room_width)

# Plot loudspeakers as blue squares
ax.scatter(room_length/2+np.sin(loudspeakers[0])*lsdist, room_width/2+np.cos(loudspeakers[0])*lsdist , marker='s', edgecolors=color_s, facecolors='none', s=100, linewidths=LW, label=r'$\mathrm{speaker\ positions}$')
for ls in loudspeakers[1:]:
    ax.scatter(room_length/2+np.sin(ls)*lsdist, room_width/2+np.cos(ls)*lsdist , marker='s', edgecolors=color_s, facecolors='none', s=100, linewidths=LW)

t = np.linspace(0,2*np.pi, 1000)
x = np.linspace(-1,1, 1000)
rad = 0.3
ax.plot(rad*np.cos(t)+room_length/2, rad*np.sin(t)+room_width/2, c=color_h, linewidth=LW)
ax.plot(x*rad/4+room_length/2, -abs(x)*rad/2+room_width/2+1.5*rad, c=color_h, linewidth=LW)

plt.text(2.65, 2.35, r'$\mathrm{dummy\ head}$', fontsize=FS, c=color_h)

# Plot microphones as unfilled red circles
ax.scatter(room_length/2+dist*microphones_x[0], room_width/2+dist*microphones_y[0], marker='o', edgecolors=color_m, facecolors='none', s=50, linewidths=LW, label=r'$\mathrm{microphones}$')
for mic_x, mic_y in zip(microphones_x[1:], microphones_y[1:]):
    ax.scatter(room_length/2+dist*mic_x, room_width/2+dist*mic_y, marker='o', edgecolors=color_m, facecolors='none', s=50, linewidths=LW)

size = 0.25
for rad in [1,2,3]:
    ax.plot(rad*size*np.cos(t), rad*size*np.sin(t), c=color_n, linewidth=LW)
    ax.plot(rad*size*np.cos(t)+room_length, rad*size*np.sin(t), c=color_n, linewidth=LW)
    ax.plot(rad*size*np.cos(t), rad*size*np.sin(t)+room_width, c=color_n, linewidth=LW)
    ax.plot(rad*size*np.cos(t)+room_length, rad*size*np.sin(t)+room_width, c=color_n, linewidth=LW)

# Set aspect ratio to equal for equal visual increments
ax.set_aspect('equal')

# Make plot boundaries thicker
for spine in ax.spines.values():
    spine.set_linewidth(LW)

# Set larger font size for labels
plt.xlabel(r'$\mathrm{room\ length\ [m]}$', fontsize=FS)
plt.ylabel(r'$\mathrm{room\ width\ [m]}$', fontsize=FS)
#plt.title('2D View of Room from Above', fontsize=16)
plt.grid(True)
plt.tick_params(labelsize=FS)  # Adjust font size for axis tick labels
plt.legend(fontsize=FS, loc='center left', bbox_to_anchor=(1, 0.5))
plt.text(0.75, 5.5, r'$\mathrm{noise}$', fontsize=FS, c=color_n)
# Display the plot
plt.show()

# Save the figure with a transparent background
fig.savefig('room_plot.png', transparent=True, bbox_inches='tight')


