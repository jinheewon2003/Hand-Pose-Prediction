import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import subprocess

# Define the connections between points based on the 21 keypoints model
connections = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]
# Define connections within the palm to be colored differently
special_connections_red = [
    (0, 1), (0, 5), (0, 17)
]
special_connections_purple = [
    (5, 9), (9, 13), (13, 17)
]

# Define colors for different sets of connections
color_special_red = 'red'
color_special_purple = 'purple'
color_normal = 'blue'

def data_visualization(data, file_to_save = None, preview = False, speed = 500, x_lim = 20.0, y_lim = 20.0, z_lim = 20.0):
    # Reshape data to (_, 21, 3) to split xyz values
    data = data.reshape(-1, 21, 3)

    # Set up figure to plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.view_init(elev=20, azim=45)  # Set view to top left corner

    lines = [ax.plot([], [], [])[0] for _ in range(len(connections))]

    # Initialize animation
    def init():
        for line in lines:
            line.set_data([], [])
            line.set_3d_properties([])
        return lines

    # Update animation
    def update(frame):
        for i, connection in enumerate(connections):
            start = data[frame, connection[0], :]
            end = data[frame, connection[1], :]
            # Unpack x, y, z coordinates
            x, y, z = zip(start, end)
            # Determine color based on connection type
            if connection in special_connections_red:
                color = color_special_red
            elif connection in special_connections_purple:
                color = color_special_purple
            else:
                color = color_normal
            lines[i].set_data(x, y)
            lines[i].set_3d_properties(z)
            lines[i].set_color(color)
        
        # Find the limits for x, y, and z coordinates to resize
        # x = data[frame, :, 0]
        # y = data[frame, :, 1]
        # z = data[frame, :, 2]
        # x_center = np.mean(x)
        # y_center = np.mean(y)
        # z_center = np.mean(z)
        # max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2
        # ax.set_xlim(x_center - max_range, x_center + max_range)
        # ax.set_ylim(y_center - max_range, y_center + max_range)
        # ax.set_zlim(z_center - max_range, z_center + max_range)
        
        ax.set_xlim(-x_lim, x_lim)
        ax.set_ylim(-y_lim, y_lim)
        ax.set_zlim(-z_lim, z_lim)
        
        # Add frame number in the top right corner
        for text in ax.texts:
            text.remove()
        frame_text = f'Frame: {frame}'
        ax.text2D(0.95, 0.95, frame_text, transform=ax.transAxes, ha='right', va='top', fontsize=12)
        
        return lines

    # Assuming the number of frames is the number of timestamps in the data
    num_frames = data.shape[0]
    ani = FuncAnimation(fig, update, frames=num_frames, init_func=init, blit=True, interval=speed)

    # Save the animation as a video file
    if file_to_save:
        ani.save(file_to_save, writer='ffmpeg')

    # Display the animation
    if preview:
        plt.show()
        
    # fig.close();

def combine_videos(video1_path, video2_path, output_path):
    ffmpeg_command = [
        'ffmpeg',
        '-i', video1_path,
        '-i', video2_path,
        '-filter_complex', '[0:v][1:v]hstack=inputs=2[v]',
        '-map', '[v]',
        output_path
    ]
    
    subprocess.run(ffmpeg_command, check=True)
    
    
def visualize_slider(data, hand_dims=[0, 3, 6, 8.5], epsilon = 15, file_to_save=None, preview=False, speed=500, x_lim=5.0, y_lim=10.0, z_lim=5.0):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim([-x_lim, x_lim])
    ax.set_ylim([0, y_lim])
    ax.set_zlim([0, z_lim])

    # Start and end points
    A = np.array([0, hand_dims[0], 0])
    B = np.array([0, hand_dims[1], 0])
    C = np.array([0, hand_dims[2], 0])
    D = np.array([0, hand_dims[3], 0])
    
    ax.view_init(elev=90, azim=0)  # Set view to top left corner

    # Plot the stick
    ax.plot([A[0], B[0], C[0], D[0]], [A[1], B[1], C[1], D[1]], [A[2], B[2], C[2], D[2]], 'k--')

    # Plot red dot at the end
    dot, = ax.plot([], [], [], 'ro')

    def update(frame):
        sorted_indices = np.argsort(data[frame])
        min_indices = sorted_indices[:2]  
        close_points = [min(min_indices), max(min_indices)]
        
        if data[frame][close_points[0]] + data[frame][close_points[1]] < hand_dims[close_points[1]] - hand_dims[close_points[0]] + epsilon:
            location = hand_dims[close_points[0]] + (data[frame][close_points[0]]) / (data[frame][close_points[0]] + data[frame][close_points[1]]) * (hand_dims[close_points[1]] - hand_dims[close_points[0]])
            dot.set_data([0], [location])
            dot.set_3d_properties([0])
        else:
            dot.set_data([], [])
            dot.set_3d_properties([])
            
        return dot,

    # Create animation
    anim = FuncAnimation(fig, update, frames=len(data), interval=speed)

    if preview:
        plt.show()

    if file_to_save:
        anim.save(file_to_save, writer='ffmpeg')