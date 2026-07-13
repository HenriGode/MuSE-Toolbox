


#%%
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Computer Modern'

colors = sns.color_palette("colorblind").as_hex()


sigs = torch.cat([signal_obj.signal_components[:].sum(dim=0, keepdim=True), signal_obj.signal_components[:]], dim=0)[:,1,:10*16000]

LW = 1.5
FS = 10

fig, (ax, ax2) = plt.subplots(2, 1, sharex=False, dpi=1200, gridspec_kw={'hspace': -0.3}, figsize=(5.5, 4))
labels = ['mic. signal', 'noise', 'source 1', 'source 2', 'source 3']
t = np.linspace(0,10,10*16000)
for idx, sig in enumerate(sigs):
    ax.plot(t, 12*sig.cpu().numpy()-idx, linewidth=LW, label=labels[idx], c=colors[idx])

ax.plot([1,1],[-10,10], c='black', linewidth=LW)
ax.plot([4,4],[-10,10], c='black', linewidth=LW)
ax.plot([7,7],[-10,10], c='black', linewidth=LW)

for spine in ax.spines.values():
    spine.set_linewidth(LW)
    
ax.set_xlim(0, 10)
ax.set_ylim(-5, 1)

ax.set_aspect(1/2.5)

textxpos = -5.5
ax.legend(fontsize=FS, loc='upper left', bbox_to_anchor=(1, 1))
ax.text(0.5, textxpos, 'noise-only', fontsize=FS, ha='center', va='center')
ax.text(2.5, textxpos, 'single-source', fontsize=FS, ha='center', va='center')
ax.text(5.5, textxpos, 'dual-source', fontsize=FS, ha='center', va='center')
ax.text(8.5, textxpos, 'triple-source', fontsize=FS, ha='center', va='center')

# Display the plot

ax.set_xticks([])
ax.set_yticks([])

scvec = np.zeros(sigs.shape[-1])
scincreases = framework.sampling_frequency * np.array(start_times)
for ts in scincreases[:-1]:
    scvec[int(ts):] += 1
ax2.plot(t, scvec, c=colors[5], linewidth=LW, label=r'source /n count')

for spine in ax2.spines.values():
    spine.set_linewidth(LW)

ax2.set_xlim(0,10)
ax2.set_aspect(1/1.3)
ax2.set_yticks([0,1,2,3])
ax2.set_xticks([])
plt.xlabel('Time')
plt.ylabel('Source Count')

#plt.tight_layout()
plt.show()
plt.tight_layout()
# Save the figure with a transparent background
fig.savefig('scenario_plot.png', transparent=True, bbox_inches='tight')






























#%%

import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
import numpy as np
from math4torch import *
import scipy

import ipywidgets as widgets
from IPython.display import clear_output
from IPython.display import display

sliderA = widgets.FloatSlider(value=27, min=0, max=360, step=1, description='Angle A')
sliderB = widgets.FloatSlider(value=165, min=0, max=360, step=1, description='Angle B')
sliderC = widgets.FloatSlider(value=315, min=0, max=360, step=1, description='Angle C')

from matplotlib.patches import FancyArrowPatch

def draw_arrow(ax, start, end, color):
    """
    Draws an arrow using FancyArrowPatch, which aligns the arrow tip with the end point.

    Args:
    ax (matplotlib.axes.Axes): The axes to draw on.
    start (tuple): The starting point (x, y) of the arrow.
    end (tuple): The ending point (x, y) of the arrow, where the tip will point.
    color (str): The color of the arrow.
    """
    arrow = FancyArrowPatch(start, end, color=color, arrowstyle='-|>', mutation_scale=25)
    ax.add_patch(arrow)

    
def parallel_proj(vector):
    return vector @ vector.T / (vector.T @ vector) 

def orthogonal_proj(vector):
    return np.identity(vector.shape[0]) - parallel_proj(vector)

def oblique_proj(vector_a, vector_b):
    return vector_a @ scipy.linalg.inv(vector_a.T @ orthogonal_proj(vector_b) @ vector_a) @ vector_a.T @ orthogonal_proj(vector_b) 

def oblique_proj(vector_a, vector_b):
    return vector_a @ scipy.linalg.inv(vector_a.T @ orthogonal_proj(vector_b) @ vector_a) @ vector_a.T @ orthogonal_proj(vector_b) 


def rotate_vector_90_deg(vector):
    """
    Rotates a 2D vector by 90 degrees counterclockwise.

    Args:
    vector (np.array): A 2D vector represented as a numpy array [x, y].

    Returns:
    np.array: The rotated vector.
    """
    rotation_matrix = np.array([[0, -1], 
                                [1,  0]])
    return np.dot(rotation_matrix, vector)

# Function to adjust label position
def adjust_label_position(vector, label_offset=[0.0625,0.0625]):
    return vector + label_offset[0] * vector/scipy.linalg.norm(vector) + label_offset[1] * rotate_vector_90_deg(vector)/scipy.linalg.norm(vector)

# Drawing vectors and positioning labels next to the tip


vector_a = np.array([[1,0]]).T
vector_a = vector_a / scipy.linalg.norm(vector_a)

vector_b = np.array([[1,1]]).T
vector_b = vector_b / scipy.linalg.norm(vector_b)

vector_c = np.array([[1,0]]).T
vector_c = vector_c / scipy.linalg.norm(vector_c)


def on_slider_change(A,B,C):
    #clear_output(wait=True)
    # Close all opened figures
    plt.close('all')
    
    
    vector_a = np.array([[np.cos(np.deg2rad(sliderA.value)),np.sin(np.deg2rad(sliderA.value))]]).T
    vector_b = np.array([[np.cos(np.deg2rad(sliderB.value)),np.sin(np.deg2rad(sliderB.value))]]).T
    vector_c = np.array([[np.cos(np.deg2rad(sliderC.value)),np.sin(np.deg2rad(sliderC.value))]]).T

    paPAC = parallel_proj(vector_a) @ vector_c
    paPBC = parallel_proj(vector_b) @ vector_c
    orPAC = orthogonal_proj(vector_a) @ vector_c
    orPBC = orthogonal_proj(vector_b) @ vector_c
    obPABC = oblique_proj(vector_a, vector_b) @ vector_c
    obPBAC = oblique_proj(vector_b, vector_a) @ vector_c
    fig, ax = plt.subplots(dpi=600)
    ax.clear()

    t = np.deg2rad(np.linspace(0,360,10000))
    ax.plot(np.cos(t),np.sin(t), color='black', linewidth=0.5)
    
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)

    # Defining colors
    color_a = 'darkorange'  # Bright color
    color_c = 'navy'        # Dark color
    color_b = 'seagreen'    # Medium brightness color
    color_c1 = 'firebrick'    # Medium brightness color

    # Setting the aspect of the plot to be equal
    ax.set_aspect('equal', adjustable='box')

    # Adding grid
    ax.grid(True)

    # Setting labels for x and y axes
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')

    # Title of the plot
    ax.set_title('Projection Operators')

    ax.plot(np.array([-5,5])*rotate_vector_90_deg(vector_a)[0]+vector_c[0], np.array([-5,+5])*rotate_vector_90_deg(vector_a)[1]+vector_c[1], linestyle=':', color=color_a)
    ax.plot(np.array([-5,5])*vector_a[0], np.array([-5,+5])*vector_a[1], linestyle=':', color=color_a)
    ax.plot(np.array([-5,5])*rotate_vector_90_deg(vector_b)[0], np.array([-5,+5])*rotate_vector_90_deg(vector_b)[1], linestyle=':', color=color_b)
    ax.plot(np.array([-5,5])*vector_b[0]+vector_c[0], np.array([-5,+5])*vector_b[1]+vector_c[1], linestyle=':', color=color_b)

    factor = 1.03
    
    # Vector A (in red)
    draw_arrow(ax, (0,0), (factor*vector_a[0,0], factor*vector_a[1,0]), color=color_a)
    label_pos_a = adjust_label_position(factor*vector_a)
    ax.text(label_pos_a[0,0], label_pos_a[1,0], r'$\mathbf{A}$', color=color_a, fontsize=12, horizontalalignment='center', verticalalignment='center')

    # Vector B (in blue)
    draw_arrow(ax, (0,0), (factor*vector_b[0,0], factor*vector_b[1,0]), color=color_b)
    label_pos_b = adjust_label_position(factor*vector_b)
    ax.text(label_pos_b[0,0], label_pos_b[1,0], r'$\mathbf{B}$', color=color_b, fontsize=12, horizontalalignment='center', verticalalignment='center')

    # Vector C (in green)
    draw_arrow(ax, (0,0), (factor*vector_c[0,0], factor*vector_c[1,0]), color=color_c1)
    label_pos_c = adjust_label_position(factor*vector_c)
    ax.text(label_pos_c[0,0], label_pos_c[1,0], r'$\mathbf{C}$', color=color_c1, fontsize=12, horizontalalignment='center', verticalalignment='center')


    draw_arrow(ax, (0,0), (factor*paPAC[0,0], factor*paPAC[1,0]), color=color_c)
    label_pos_a = adjust_label_position(paPAC)
    ax.text(label_pos_a[0,0], label_pos_a[1,0], r'$\mathbf{P}^{\parallel}_{\mathbf{A}} \mathbf{C}$', color=color_c, fontsize=12, horizontalalignment='center', verticalalignment='center')


    # Vector C (in green)
    draw_arrow(ax, (0,0), (factor*obPABC[0,0], factor*obPABC[1,0]), color=color_c)
    label_pos_c = adjust_label_position(obPABC)
    ax.text(label_pos_c[0,0], label_pos_c[1,0], r'$\mathbf{P}^{\angle}_{\mathbf{AB}} \mathbf{C}$', color=color_c, fontsize=12, horizontalalignment='center', verticalalignment='center')

    # Vector B (in blue)
    draw_arrow(ax, (0,0), (factor*orPBC[0,0], factor*orPBC[1,0]), color=color_c)
    label_pos_b = adjust_label_position(orPBC)
    ax.text(label_pos_b[0,0], label_pos_b[1,0], r'$\mathbf{P}^{\perp}_{\mathbf{B}} \mathbf{C}$', color=color_c, fontsize=12, horizontalalignment='center', verticalalignment='center')

    if False:
        
        ax.plot(np.array([-5,5])*vector_a[0]+vector_c[0], np.array([-5,+5])*vector_a[1]+vector_c[1], linestyle=':', color=color_a)
        ax.plot(np.array([-5,5])*rotate_vector_90_deg(vector_a)[0], np.array([-5,+5])*rotate_vector_90_deg(vector_a)[1], linestyle=':', color=color_a)
        ax.plot(np.array([-5,5])*rotate_vector_90_deg(vector_b)[0]+vector_c[0], np.array([-5,+5])*rotate_vector_90_deg(vector_b)[1]+vector_c[1], linestyle=':', color=color_b)
        ax.plot(np.array([-5,5])*vector_b[0], np.array([-5,+5])*vector_b[1], linestyle=':', color=color_b)

        
        # Vector C (in green)
        draw_arrow(ax, (0,0), (factor*obPBAC[0,0], factor*obPBAC[1,0]), color=color_c)
        label_pos_c = adjust_label_position(obPBAC)
        ax.text(label_pos_c[0,0], label_pos_c[1,0], r'$\mathbf{P}^{\angle}_{\mathbf{BA}} \mathbf{C}$', color=color_c, fontsize=12, horizontalalignment='center', verticalalignment='center')
        
        draw_arrow(ax, (0,0), (factor*paPBC[0,0], factor*paPBC[1,0]), color=color_c)
        label_pos_a = adjust_label_position(paPBC)
        ax.text(label_pos_a[0,0], label_pos_a[1,0], r'$\mathbf{P}^{\parallel}_{\mathbf{B}} \mathbf{C}$', color=color_c, fontsize=12, horizontalalignment='center', verticalalignment='center')

        # Vector B (in blue)
        draw_arrow(ax, (0,0), (factor*orPAC[0,0], factor*orPAC[1,0]), color=color_c)
        label_pos_b = adjust_label_position(orPAC)
        ax.text(label_pos_b[0,0], label_pos_b[1,0], r'$\mathbf{P}^{\perp}_{\mathbf{A}} \mathbf{C}$', color=color_c, fontsize=12, horizontalalignment='center', verticalalignment='center')
   

    # Display the plot with adjusted label positions
    plt.show()
    plt.savefig('proj_op_scheme.png')
    fig.canvas.draw_idle()
    
    

widgets.interactive(on_slider_change, A=sliderA, B=sliderB, C=sliderC)
#sliderB.observe(on_slider_change, names='value')
#sliderC.observe(on_slider_change, names='value')




















# %%
#

import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
import numpy as np
from math4torch import *
import scipy
import seaborn as sns

Avalue=27
Bvalue=165
Cvalue=315

from matplotlib.patches import FancyArrowPatch

def draw_arrow(ax, start, end, color):
    """
    Draws an arrow using FancyArrowPatch, which aligns the arrow tip with the end point.

    Args:
    ax (matplotlib.axes.Axes): The axes to draw on.
    start (tuple): The starting point (x, y) of the arrow.
    end (tuple): The ending point (x, y) of the arrow, where the tip will point.
    color (str): The color of the arrow.
    """
    arrow = FancyArrowPatch(start, end, color=color, arrowstyle='-|>', mutation_scale=25)
    ax.add_patch(arrow)

    
def parallel_proj(vector):
    return vector @ vector.T / (vector.T @ vector) 

def orthogonal_proj(vector):
    return np.identity(vector.shape[0]) - parallel_proj(vector)

def oblique_proj(vector_a, vector_b):
    return vector_a @ scipy.linalg.inv(vector_a.T @ orthogonal_proj(vector_b) @ vector_a) @ vector_a.T @ orthogonal_proj(vector_b) 

def oblique_proj(vector_a, vector_b):
    return vector_a @ scipy.linalg.inv(vector_a.T @ orthogonal_proj(vector_b) @ vector_a) @ vector_a.T @ orthogonal_proj(vector_b) 


def rotate_vector_90_deg(vector):
    """
    Rotates a 2D vector by 90 degrees counterclockwise.

    Args:
    vector (np.array): A 2D vector represented as a numpy array [x, y].

    Returns:
    np.array: The rotated vector.
    """
    rotation_matrix = np.array([[0, -1], 
                                [1,  0]])
    return np.dot(rotation_matrix, vector)

# Function to adjust label position
def adjust_label_position(vector, label_offset=[0.0625,0.0625]):
    return vector + label_offset[0] * vector/scipy.linalg.norm(vector) + label_offset[1] * rotate_vector_90_deg(vector)/scipy.linalg.norm(vector)

# Drawing vectors and positioning labels next to the tip


vector_a = np.array([[1,0]]).T
vector_a = vector_a / scipy.linalg.norm(vector_a)

vector_b = np.array([[1,1]]).T
vector_b = vector_b / scipy.linalg.norm(vector_b)

vector_c = np.array([[1,0]]).T
vector_c = vector_c / scipy.linalg.norm(vector_c)


#clear_output(wait=True)
# Close all opened figures
plt.close('all')


vector_a = np.array([[np.cos(np.deg2rad(Avalue)),np.sin(np.deg2rad(Avalue))]]).T
vector_b = np.array([[np.cos(np.deg2rad(Bvalue)),np.sin(np.deg2rad(Bvalue))]]).T
vector_c = np.array([[np.cos(np.deg2rad(Cvalue)),np.sin(np.deg2rad(Cvalue))]]).T

paPAC = parallel_proj(vector_a) @ vector_c
paPBC = parallel_proj(vector_b) @ vector_c
orPAC = orthogonal_proj(vector_a) @ vector_c
orPBC = orthogonal_proj(vector_b) @ vector_c
obPABC = oblique_proj(vector_a, vector_b) @ vector_c
obPBAC = oblique_proj(vector_b, vector_a) @ vector_c
fig, ax = plt.subplots(dpi=600)
ax.clear()

t = np.deg2rad(np.linspace(0,360,10000))
ax.plot(np.cos(t),np.sin(t), color='black', linewidth=0.5)

ax.set_xlim(-1.2, 1.2)
ax.set_ylim(-1.2, 1.2)

# Defining colors
colors = sns.color_palette("colorblind").as_hex()
color_a = 'darkorange'  # Bright color
color_c = 'navy'        # Dark color
color_b = 'seagreen'    # Medium brightness color
color_c1 = 'firebrick'    # Medium brightness color
color_a = colors[1]  # Bright color
color_c = colors[0]  #'navy'        # Dark color
color_b = colors[2]  #'seagreen'    # Medium brightness color
color_c1 = colors[3]  #'firebrick'    # Medium brightness color


# Setting the aspect of the plot to be equal
ax.set_aspect('equal', adjustable='box')

# Adding grid
ax.grid(True)

# Setting labels for x and y axes
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')

# Title of the plot
ax.set_title(r'$\mathrm{Projection\ Operators}$')


factor = 1.03

# Vector A (in red)
draw_arrow(ax, (0,0), (factor*vector_a[0,0], factor*vector_a[1,0]), color=color_a)
label_pos_a = adjust_label_position(factor*vector_a)
ax.text(label_pos_a[0,0], label_pos_a[1,0], r'$\mathbf{A}$', color=color_a, fontsize=12, horizontalalignment='center', verticalalignment='center')

# Vector B (in blue)
draw_arrow(ax, (0,0), (factor*vector_b[0,0], factor*vector_b[1,0]), color=color_b)
label_pos_b = adjust_label_position(factor*vector_b)
ax.text(label_pos_b[0,0], label_pos_b[1,0], r'$\mathbf{B}$', color=color_b, fontsize=12, horizontalalignment='center', verticalalignment='center')

# Vector C (in green)
draw_arrow(ax, (0,0), (factor*vector_c[0,0], factor*vector_c[1,0]), color=color_c1)
label_pos_c = adjust_label_position(factor*vector_c)
ax.text(label_pos_c[0,0], label_pos_c[1,0], r'$\mathbf{C}$', color=color_c1, fontsize=12, horizontalalignment='center', verticalalignment='center')


draw_arrow(ax, (0,0), (factor*paPAC[0,0], factor*paPAC[1,0]), color=color_c)
label_pos_a = adjust_label_position(paPAC)
ax.text(label_pos_a[0,0], label_pos_a[1,0], r'$\mathbf{P}^{\parallel}_{\mathbf{A}} \mathbf{C}$', color=color_c, fontsize=12, horizontalalignment='center', verticalalignment='center')


# Vector C (in green)
draw_arrow(ax, (0,0), (factor*obPABC[0,0], factor*obPABC[1,0]), color=color_c)
label_pos_c = adjust_label_position(obPABC)
ax.text(label_pos_c[0,0], label_pos_c[1,0], r'$\mathbf{P}^{\angle}_{\mathbf{AB}} \mathbf{C}$', color=color_c, fontsize=12, horizontalalignment='center', verticalalignment='center')

# Vector B (in blue)
draw_arrow(ax, (0,0), (factor*orPBC[0,0], factor*orPBC[1,0]), color=color_c)
label_pos_b = adjust_label_position(orPBC)
ax.text(label_pos_b[0,0], label_pos_b[1,0], r'$\mathbf{P}^{\perp}_{\mathbf{B}} \mathbf{C}$', color=color_c, fontsize=12, horizontalalignment='center', verticalalignment='center')


ax.plot(np.array([-5,5])*rotate_vector_90_deg(vector_a)[0]+vector_c[0], np.array([-5,+5])*rotate_vector_90_deg(vector_a)[1]+vector_c[1], linestyle=':', color=color_a)
ax.plot(np.array([-5,5])*vector_a[0], np.array([-5,+5])*vector_a[1], linestyle=':', color=color_a)
ax.plot(np.array([-5,5])*rotate_vector_90_deg(vector_b)[0], np.array([-5,+5])*rotate_vector_90_deg(vector_b)[1], linestyle=':', color=color_b)
ax.plot(np.array([-5,5])*vector_b[0]+vector_c[0], np.array([-5,+5])*vector_b[1]+vector_c[1], linestyle=':', color=color_b)


if False:
    
    ax.plot(np.array([-5,5])*vector_a[0]+vector_c[0], np.array([-5,+5])*vector_a[1]+vector_c[1], linestyle=':', color=color_a)
    ax.plot(np.array([-5,5])*rotate_vector_90_deg(vector_a)[0], np.array([-5,+5])*rotate_vector_90_deg(vector_a)[1], linestyle=':', color=color_a)
    ax.plot(np.array([-5,5])*rotate_vector_90_deg(vector_b)[0]+vector_c[0], np.array([-5,+5])*rotate_vector_90_deg(vector_b)[1]+vector_c[1], linestyle=':', color=color_b)
    ax.plot(np.array([-5,5])*vector_b[0], np.array([-5,+5])*vector_b[1], linestyle=':', color=color_b)

    
    # Vector C (in green)
    draw_arrow(ax, (0,0), (factor*obPBAC[0,0], factor*obPBAC[1,0]), color=color_c)
    label_pos_c = adjust_label_position(obPBAC)
    ax.text(label_pos_c[0,0], label_pos_c[1,0], r'$\mathbf{P}^{\angle}_{\mathbf{BA}} \mathbf{C}$', color=color_c, fontsize=12, horizontalalignment='center', verticalalignment='center')
    
    draw_arrow(ax, (0,0), (factor*paPBC[0,0], factor*paPBC[1,0]), color=color_c)
    label_pos_a = adjust_label_position(paPBC)
    ax.text(label_pos_a[0,0], label_pos_a[1,0], r'$\mathbf{P}^{\parallel}_{\mathbf{B}} \mathbf{C}$', color=color_c, fontsize=12, horizontalalignment='center', verticalalignment='center')

    # Vector B (in blue)
    draw_arrow(ax, (0,0), (factor*orPAC[0,0], factor*orPAC[1,0]), color=color_c)
    label_pos_b = adjust_label_position(orPAC)
    ax.text(label_pos_b[0,0], label_pos_b[1,0], r'$\mathbf{P}^{\perp}_{\mathbf{A}} \mathbf{C}$', color=color_c, fontsize=12, horizontalalignment='center', verticalalignment='center')


#plt.legend(['unit  circle', 'vector A', 'vector B', 'vector C', 'projection lines'])

# Display the plot with adjusted label positions

plt.savefig('proj_op_scheme.png')
plt.show()
    


# %%
