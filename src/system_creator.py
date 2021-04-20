class SystemCreator(object):
    """
    Represents a system with boundaries and inward vectors. Verifying that it is up to the user.
    TODO : It is more important for the potential influx and outflux if you use some.
    
    """
    def __init__(self, segments):
        """ Initialize a system from a list of segments.

        Args:
            segments (list of Segment): the list containing all the segments of the system. 
        """
        self.segments = segments 
        self.min_x, self.max_x, self.min_y, self.max_y = self._init_extremal_values()

    def _init_extremal_values(self):
        segment_x_list = []
        segment_y_list = []
        for segment in self.segments:
            p1, p2 = segment.get_p1(), segment.get_p2()
            segment_x_list.append(p1.x)
            segment_x_list.append(p2.x)
            segment_y_list.append(p1.y)
            segment_y_list.append(p2.y)
        max_x, min_x = max(segment_x_list), min(segment_x_list)
        max_y, min_y = max(segment_y_list), min(segment_y_list)
        return min_x, max_x, min_y, max_y

    # -------------------------- Getter / Setter --------------- #

    def get_size(self):
        return self.max_x - self.min_x, self.max_y - self.min_y

    def get_extremal_values(self):
        return {
            'min_x' : self.min_x,
            'max_x' : self.max_x,
            'min_y' : self.min_y,
            'max_y' : self.max_y
        }

    def get_segments(self):
        return self.segments
    


    