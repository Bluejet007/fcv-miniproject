from multiprocessing import Process, Queue
import numpy as np
import OpenGL.GL as gl
import pypangolin as pangolin

# Adding G2O's path
import os
import sys
os.add_dll_directory(r"D:\MAHE\5thSem\FCVLab\SLAM\Gitstuff\g2opy\bin\Release")
sys.path.append(r"D:\MAHE\5thSem\FCVLab\SLAM\Gitstuff\g2opy\bin\Release")
import g2o

# Draws a camera at pose T (4x4 transformation matrix)
def draw_camera(T, scale=1):
    # Define camera corners in camera frame
    pts = np.array([
        [0, 0, 0],      # Centre
        [0.5, 0.5, 1],  # Top-right
        [0.5, -0.5, 1], # Bottom-right
        [-0.5,-0.5, 1], # Bottom-left
        [-0.5, 0.5, 1]  # Top-left
    ], dtype=np.float32) * scale

    # Transform to world frame (inverse exttrinsic matrix)
    pts_h = np.hstack([pts, np.ones((5,1))])
    pts_w = (T @ pts_h.T).T[:, :3]

    # Lines connecting camera center to corners
    lines = [
        (0, 1), (0, 2), (0, 3), (0, 4),
        (1, 2), (2, 3), (3, 4), (4, 1)
    ]
    for i, j in lines:
        pangolin.glDrawLines([pts_w[i].reshape(3, 1), pts_w[j].reshape(3, 1)])

# A point is a 3D point in the world
# Each point is observed in multiple Frames
class Point:
    def __init__(self, mapp, loc):
        self.pt = loc
        self.frames = []
        self.idxs = []
        
        self.id = len(mapp.points)
        mapp.points.append(self)

    def add_observation(self, frame, idx):
        frame.pts[idx] = self
        self.frames.append(frame)
        self.idxs.append(idx)

class Descriptor:
    def __init__(self, W, H, K):
        self.frames = []
        self.points = []
        self.state = None
        self.W = W
        self.H = H
        self.K = K
        self.q = None

    # G2O optimisation
    def optimise(self, max_iterations=10):
        optimiser = g2o.SparseOptimizer()
        solver = g2o.BlockSolverSE3(g2o.LinearSolverCSparseSE3())
        solver = g2o.OptimizationAlgorithmLevenberg(solver)
        optimiser.set_algorithm(solver)

        # Add frame vertices
        for i, frame in enumerate(self.frames):
            v_se3 = g2o.VertexSE3Expmap()
            v_se3.set_id(i)
            if i == 0:
                v_se3.set_fixed(True)  # Fix the 1st frame as reference
                return
            v_se3.set_estimate(g2o.SE3Quat(frame.pose[:3, :3], frame.pose[:3, 3]))
            optimiser.add_vertex(v_se3)

        # Add relative pose edges between consecutive frames
        for i in range(1, len(self.frames)):
            prev_frame = self.frames[i - 1]
            curr_frame = self.frames[i]

            # Relative transform between the 2 frames
            rel = np.linalg.inv(prev_frame.pose) @ curr_frame.pose
            R = rel[:3, :3]
            t = rel[:3, 3]

            edge = g2o.EdgeSE3()
            edge.set_vertex(0, optimiser.vertex(i - 1))
            edge.set_vertex(1, optimiser.vertex(i))
            edge.set_measurement(g2o.SE3Quat(R, t))
            edge.set_information(np.eye(6))  # Identity info matrix
            optimiser.add_edge(edge)

        # Run the optimisation
        optimiser.initialize_optimization()
        optimiser.optimize(max_iterations)

        # Update poses
        for i, frame in enumerate(self.frames):
            est = optimiser.vertex(i).estimate()
            frame.pose[:3, :3] = est.rotation().matrix()
            frame.pose[:3, 3] = est.translation()

    def create_viewer(self):
      self.q = Queue()
      self.vp = Process(target=self.viewer_thread, args=(self.q, ))
      self.vp.daemon = True
      self.vp.start()

    def viewer_thread(self, q):
        self.viewer_init(1024, 768)
        while True:
            self.viewer_refresh(q)

    def viewer_init(self, w, h):
        pangolin.CreateWindowAndBind('Main', w, h)
        gl.glEnable(gl.GL_DEPTH_TEST)

        self.scam = pangolin.OpenGlRenderState(
        pangolin.ProjectionMatrix(w, h, 420, 420, w // 2, h // 2, 0.2, 10000),
        pangolin.ModelViewLookAt(0, -10, -8,
                                 0, 0, 0,
                                 0, -1, 0))
        self.handler = pangolin.Handler3D(self.scam)

        # Create interactive view
        self.dcam = pangolin.CreateDisplay()
        self.dcam.SetBounds(pangolin.Attach(0.0), pangolin.Attach(1.0), pangolin.Attach(0.0), pangolin.Attach(1.0), -w/h)
        self.dcam.SetHandler(self.handler)

    def viewer_refresh(self, q):
        if self.state is None or not q.empty():
            self.state = q.get()

        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glClearColor(0, 0, 0, 0)
        self.dcam.Activate(self.scam)

        # Draw keypoints
        gl.glPointSize(2)
        gl.glColor3f(0.184314, 0.309804, 0.184314)
        points = np.array(self.state[1][:, :3] + 1, dtype=np.float32)
        points = points.T
        points = np.ascontiguousarray(points, dtype=np.float32)
        points_to_draw = [p.reshape(-1, 1) for p in points.T]
        pangolin.glDrawPoints(points_to_draw)

        # Draw poses
        gl.glColor3f(0.0, 1.0, 1.0)
        for T in self.state[0]: # state[0] is a list of 4x4 poses
            draw_camera(T)

        pangolin.FinishFrame()

    def display(self):
        if self.q is None:
            return
        poses, pts = [], []
        for f in self.frames:
            poses.append(f.pose)
        for p in self.points:
            pts.append(p.pt)
        self.q.put((np.array(poses), np.array(pts)))