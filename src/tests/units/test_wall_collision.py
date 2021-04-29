from src.utils.wall_collision import _collision_with_wall, _handler_wall_collision, _reflect_particle

import unittest
import numpy as np

import matplotlib.pyplot as plt


class TestSystem(unittest.TestCase):
    # TODO : correct it 
    # TODO : make it all < eps

    def plot_system(self, segments):
        for segment in segments :
            p1, p2 = segment.get_p1(), segment.get_p2()
            plt.plot([p1.x, p2.x], [p1.y, p2.y], color = 'k')
            
    def plot_part(self, p1, p2, v1, v2, r):
        plt.plot([p1.x, p2.x], [p1.y, p2.y], '-', color = 'b')
        plt.arrow(x = p1.x, y = p1.y, dx = v1.x*0.1, dy = v1.y*0.1, color = 'r')
        plt.arrow(x = p2.x, y = p2.y, dx = v2.x*0.1, dy = v2.y*0.1, color = 'r')

    def test_wall_collision(self):
        visual_debug = False

        p1 = Vector(-1,-1)
        p2 = Vector(1,-1)
        p3 = Vector(1,1)
        p4 = Vector(-1,1)
        
        segments = [Segment(p1,p2),
            Segment(p2,p3),
            Segment(p3,p4),
            Segment(p4,p1)] # square centered in (0,0) with lenght = 2

        radius = 0.1

        eps = 0.001
        # simple cases
            # 1
        pos = Vector(2,0,0)
        v = Vector(1,0,0)
        min_time, idx_segment, min_pos_intersect = _collision_with_wall(pos, v, radius, segments, strategy = 'past')
        self.assertTrue(min_time == 1.1)  # (min_time - 0.9)<eps)
        self.assertTrue(idx_segment == 1)
        self.assertTrue((min_pos_intersect - Vector(0.9,0,0)).norm()<eps) # (min_pos_intersect-Vector(0.9,0,0)).norm()<eps)

        new_pos, new_v = _reflect_particle(v, min_time-2*radius/v.x, (segments[idx_segment].get_p2()-segments[idx_segment].get_p1()).normalize(), min_pos_intersect)
        self.assertTrue((new_pos-Vector(0,0,0)).norm() < eps)
        self.assertTrue((new_v+v).norm() < eps )
        
        if(visual_debug):
            fig = plt.figure(clear=True)
            self.plot_part(pos,new_pos,v,new_v,radius)
            self.plot_system(segments)
            plt.show()

            # 2
        pos = Vector(-2,0,0)
        v = Vector(-0.5,0,0)
        min_time, idx_segment, min_pos_intersect = _collision_with_wall(pos, v, radius, segments, strategy = 'past')
        self.assertTrue(min_time == 2.2)
        self.assertTrue(idx_segment == 3)
        self.assertTrue((min_pos_intersect-Vector(-0.9,0,0)).norm()<eps)

        new_pos, new_v = _reflect_particle(v, min_time-2*radius/(-v.x), (segments[idx_segment].get_p2()-segments[idx_segment].get_p1()).normalize(), min_pos_intersect)
        self.assertTrue((new_pos-Vector(0,0,0)).norm() < eps)
        self.assertTrue((new_v+v).norm() < eps )

        if(visual_debug):
            fig = plt.figure(clear=True)
            self.plot_part(pos,new_pos,v,new_v,radius)
            self.plot_system(segments)
            plt.show()

            # 3
        pos = Vector(0,2,0)
        v = Vector(0,1,0)
        min_time, idx_segment, min_pos_intersect = _collision_with_wall(pos, v, radius, segments, strategy = 'past')
        self.assertTrue(min_time == 1.1)
        self.assertTrue(idx_segment == 2)
        self.assertTrue((min_pos_intersect - Vector(0,0.9,0)).norm()<eps)

        new_pos, new_v = _reflect_particle(v, min_time-2*radius/v.y, (segments[idx_segment].get_p2()-segments[idx_segment].get_p1()).normalize(), min_pos_intersect)
        self.assertTrue((new_pos-Vector(0,0,0)).norm() < eps)
        self.assertTrue((new_v+v).norm() < eps )
        
        if(visual_debug):
            fig = plt.figure(clear=True)
            self.plot_part(pos,new_pos,v,new_v,radius)
            self.plot_system(segments)
            plt.show()

            # 4
        pos = Vector(0,-2,0)
        v = Vector(0,-1,0)
        min_time, idx_segment, min_pos_intersect = _collision_with_wall(pos, v, radius, segments, strategy = 'past')
        self.assertTrue(min_time == 1.1)
        self.assertTrue(idx_segment == 0)
        self.assertTrue((min_pos_intersect - Vector(0,-0.9,0)).norm()<eps)

        new_pos, new_v = _reflect_particle(v, min_time-2*radius/(-v.y), (segments[idx_segment].get_p2()-segments[idx_segment].get_p1()).normalize(), min_pos_intersect)
        self.assertTrue((new_pos-Vector(0,0,0)).norm() < eps)
        self.assertTrue((new_v+v).norm() < eps )

        if(visual_debug):
            fig = plt.figure(clear=True)
            self.plot_part(pos,new_pos,v,new_v,radius)
            self.plot_system(segments)
            plt.show()

            # 5
        pos = Vector(0,0,0)
        v = Vector(0,0,1)
        min_time, idx_segment, min_pos_intersect = _collision_with_wall(pos, v, radius, segments, strategy = 'past')
        #  1e15 ,-1, None (default values)
        self.assertTrue(min_time == 1e15)
        self.assertTrue(idx_segment == -1)
        self.assertTrue(min_pos_intersect is None)

        # complex ones
            # 6
        pos = Vector(2,2,0)
        v = Vector(1,1,0)
        min_time, idx_segment, min_pos_intersect = _collision_with_wall(pos, v, radius, segments, strategy = 'past')
        self.assertTrue(min_time == 1.1)
        self.assertTrue(idx_segment == 1) # could be 2 too, however the algorithm chooses the first one encountered.
        self.assertTrue((min_pos_intersect - Vector(0.9,0.9)).norm()<eps)

        new_pos, new_v = _reflect_particle(v, min_time-2*radius/v.x, (segments[idx_segment].get_p2()-segments[idx_segment].get_p1()).normalize(), min_pos_intersect)
        self.assertTrue((new_pos-Vector(0,1.8,0)).norm()<eps)
        self.assertTrue((new_v-Vector(-v.x,v.y,0)).norm()<eps)

        if(visual_debug):
            fig = plt.figure(clear=True)
            self.plot_part(pos,new_pos,v,new_v,radius)
            self.plot_system(segments)
            plt.show()

            # 7
        pos = Vector(1,2,0)
        v = Vector(0.5,1,0)
        min_time, idx_segment, min_pos_intersect = _collision_with_wall(pos, v, radius, segments, strategy = 'past')
        self.assertTrue(min_time == 1.1)
        self.assertTrue(idx_segment == 2)
        self.assertTrue((min_pos_intersect -Vector(0.45,0.9,0)).norm()<eps)

        new_pos, new_v = _reflect_particle(v, min_time-2*radius/v.y, (segments[idx_segment].get_p2()-segments[idx_segment].get_p1()).normalize(), min_pos_intersect)
        self.assertTrue((new_pos-Vector(0.9, 0, 0)).norm()<eps)
        self.assertTrue((new_v-Vector(v.x,-v.y,0)).norm()<eps)

        if(visual_debug):
            fig = plt.figure(clear=True)
            self.plot_part(pos,new_pos,v,new_v,radius)
            self.plot_system(segments)
            plt.show()

if __name__ == '__main__':
    unittest.main()