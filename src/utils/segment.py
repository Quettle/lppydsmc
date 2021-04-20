from src.utils.vector import Vector

class Segment(object):
    """ Class that represents a 2D-segment considering only the first two dimensions.
    """
    def __init__(self, p1, p2):
        """ Initialize a segment from two points p1 and p2. From them, a normal vector *n* is deduced such that ((p2-p1), n, (p2-p1) x n) is a direct system and n is normalized.

        Args:
            p1 (Vector, list, tuple): the first point of the segment
            p2 (Vector, list, tuple): the second point of the segment
        """

        self.p1 = self._init_point(p1)
        self.p2 = self._init_point(p2)
        try :
            assert(self.p1.dimension()>1 and self.p2.dimension()>1)
        except ValueError:
            raise ValueError(f"Dimension should be superior to two for both points. Got : p1 : {self.p1.dimension()}, p2 : {self.p2.dimension()}")
        self.n = Vector(self.p1.y - self.p2.y, self.p2.x - self.p1.x).normalize()

    def _init_point(self, p):
        if(type(p) is Vector):
            return p
        elif (type(p) in [tuple,list]): 
            return Vector(*p)
        else:
            raise TypeError(f"Point should be of either a Vector, a tuple or a list. Received : {type(p)}.")

    # ----------------------- Getter / Setter ------------------------ #

    def get_n(self):
        return self.n
    
    def get_p1(self):
        return self.p1

    def get_p2(self):
        return self.p2

    def get_segment(self):
        return self.p1, self.p2

    def invert_n(self):
        """Invert the normal vector.
        """
        self.n = -self.n

    # ----------------------- str and repr ------------------------ #
    def __str__(self):
        return f"[({self.p1}),({self.p2})], n = ({self.n})"

    def __repr__(self) -> str:
        return f"Segment({self.p1}, {self.p2})"
    
    def __eq__(self, o: object) -> bool:
        return all([self.p1 == o.get_p1(), self.p2 == o.get_p2(), self.n == o.get_n()])
        