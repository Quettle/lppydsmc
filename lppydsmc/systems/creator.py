import numpy as np

class SystemCreator(object):
    """
    Represents a system with boundaries.
    Example : 
        Boundary : type = ndarray ; value = [x1,y1,x2,y2]
    """
    def __init__(self, segments, idx_out_segments):
        """ Initialize a system from a list of segments (2D-ndarray). The segments have to be given in a clock-wise compared to the inside of the system.
        It is also the case for the extremities of a segment : (x1,y1)<(x2,y2) clock-wise.

        Args:
            segments (2D-ndarray): the list containing all the segments of the system. 
        """
        self.segments, self.a, self.n = self._init_segments(segments)
        self.min_x, self.max_x, self.min_y, self.max_y = self._init_extremal_values()
        self.idx_out_segments = idx_out_segments
    def _init_segments(self, segments):
        segments_ = []
        a = np.zeros((segments.shape[0], 3))
        normal = [] # normal vectors facing inward (that is why we can not use a to get to the normal vectors but we can use the inital segments)
                    # defined in a counter-clock wise manner
        for k, segment in enumerate(segments):
            x1, y1, x2, y2 = segment
            normal.append([y2-y1, x1-x2])
            a[k, 2] = np.linalg.norm(segment[2:]-segment[:2])
            assert((x1!=x2) or (y1!=y2))
            
            if(x1>x2 or (x1==x2 and y1>y2)):
                segments_.append([x2, y2, x1, y1])
                a[k, :2] = np.array([x1-x2, y1-y2])/a[k, 2]
            else :
                segments_.append([x1, y1, x2, y2])
                a[k, :2] = np.array([x2-x1, y2-y1])/a[k, 2]

        normal = np.array(normal)
        norm = np.linalg.norm(normal, axis = 1)
        return np.array(segments_), a, normal/np.expand_dims(norm, axis = 1)

    def _init_extremal_values(self):
        segment_x_list = []
        segment_y_list = []
        for segment in self.segments:
            x1, y1, x2, y2 = segment
            segment_x_list.append(x1)
            segment_x_list.append(x2)
            segment_y_list.append(y1)
            segment_y_list.append(y2)
        max_x, min_x = max(segment_x_list), min(segment_x_list)
        max_y, min_y = max(segment_y_list), min(segment_y_list)
        return min_x, max_x, min_y, max_y

    # -------------------------- Getter / Setter --------------- #

    def get_shape(self):
        return np.array([self.max_x - self.min_x, self.max_y - self.min_y])

    def get_extremal_values(self):
        return {
            'min_x' : self.min_x,
            'max_x' : self.max_x,
            'min_y' : self.min_y,
            'max_y' : self.max_y
        }

    def get_segments(self):
        return self.segments
    
    def get_offsets(self):
        return np.array([self.min_x, self.min_y])

    def get_dir_vects(self):
        return self.a

    def get_normal_vectors(self):
        return self.n

    def get_idx_out_segments(self):
        return self.idx_out_segments
    
    def __str__(self) -> str:
        return f'System : shape = {self.get_shape()} m - offsets = {self.get_offsets()} - {len(self.segments)} segments of which {len(self.idx_out_segments)} are exits'