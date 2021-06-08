import numpy as np

class SystemCreator(object):
    """
    Represents a system with boundaries.
    Example : 
        Boundary : type = ndarray ; value = [x1,y1,x2,y2]
    """
    def __init__(self, segments):
        """ Initialize a system from a list of segments (2D-ndarray).

        Args:
            segments (2D-ndarray): the list containing all the segments of the system. 
        """
        self.segments, self.a = self._init_segments(segments)
        self.min_x, self.max_x, self.min_y, self.max_y = self._init_extremal_values()

    def _init_segments(self, segments):
        segments_ = []
        a = np.zeros((segments.shape[0], 3))
        for k, segment in enumerate(segments):
            x1, y1, x2, y2 = segment
            a[k, 2] = np.linalg.norm(segment[2:]-segment[:2])
            assert((x1!=x2) or (y1!=y2))

            if(x1>x2 or (x1==x2 and y1>y2)):
                segments_.append([x2, y2, x1, y1])
                a[k, :2] = np.array([x1-x2, y1-y2])/a[k, 2]
            else :
                segments_.append([x1, y1, x2, y2])
                a[k, :2] = np.array([x2-x1, y2-y1])/a[k, 2]
        return np.array(segments_), a

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

    def system_shape(self):
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

    