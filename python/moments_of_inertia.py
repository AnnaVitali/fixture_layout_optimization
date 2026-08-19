import math
import numpy as np

AREA = "area"
ABSOLUTE_MOMENTS_OF_INERTIA = "absolute_moments_of_inertia"
BARICENTRIC_MOMENTS_OF_INERTIA = "baricentric_moments_of_inertia"
BARYCENTER = "barycenter"
ANGLE = "angle"


class InertiaAnalysis:
    """This class contains methods for calculating the area, absolute moments

    of inertia, baricentric moments of inertia, and the center of gravity of
    polygons.
    """

    def __init__(self) -> None:
        pass

    @staticmethod
    def _sort_vertices(vertices):
        """Ordina i vertici in senso antiorario per evitare figure intrecciate

        (clessidra) prima dell'integrazione di Green.
        """
        cx = sum(v[0] for v in vertices) / len(vertices)
        cy = sum(v[1] for v in vertices) / len(vertices)
        return sorted(
            vertices, key=lambda v: math.atan2(v[1] - cy, v[0] - cx)
        )

    @staticmethod
    def compute_polygon_area(vertices):
        """Computes the area of a polygon using the shoelace formula."""
        sorted_v = InertiaAnalysis._sort_vertices(vertices)
        x = np.array([v[0] for v in sorted_v])
        y = np.array([v[1] for v in sorted_v])
        return 0.5 * np.abs(
            np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))
        )

    @staticmethod
    def compute_absolute_moments_of_inertia(vertices):
        """Computes the absolute moments of inertia (jx, jy, jxy) of a polygon

        respect to the origin (0,0).
        """
        sorted_v = InertiaAnalysis._sort_vertices(vertices)
        n = len(sorted_v)

        jx = 0.0
        jy = 0.0
        jxy = 0.0

        for i in range(n):
            x_i, y_i = sorted_v[i]
            x_next, y_next = sorted_v[(i + 1) % n]

            common_term = x_i * y_next - x_next * y_i

            jx += (y_i**2 + y_i * y_next + y_next**2) * common_term
            jy += (x_i**2 + x_i * x_next + x_next**2) * common_term
            jxy += (
                x_i * y_next
                + 2 * x_i * y_i
                + 2 * x_next * y_next
                + x_next * y_i
            ) * common_term

        jx = abs(jx) / 12.0
        jy = abs(jy) / 12.0
        jxy = jxy / 24.0

        return jx, jy, jxy

    @staticmethod
    def compute_baricentric_moments_of_inertia(polygon):
        """Computes the baricentric moments of inertia of a polygon."""
        area = polygon[AREA]
        x_g, y_g = polygon[BARYCENTER]
        jx, jy, jxy = polygon[ABSOLUTE_MOMENTS_OF_INERTIA]
        angle_rad = polygon.get(ANGLE, 0.0)

        # Trasporto Huygens-Steiner al baricentro locale della singola figura
        i_prime = jx - y_g**2 * area
        j_prime = jy - x_g**2 * area
        ij_prime = jxy - y_g * x_g * area

        # Rotazione tensoriale facoltativa secondo l'angolo
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)

        i = (
            i_prime * cos_a**2
            + j_prime * sin_a**2
            - ij_prime * 2 * sin_a * cos_a
        )
        j = (
            i_prime * sin_a**2
            + j_prime * cos_a**2
            + ij_prime * 2 * sin_a * cos_a
        )
        ij = (i_prime - j_prime) * sin_a * cos_a + ij_prime * (
            cos_a**2 - sin_a**2
        )
        
        #print("I_prime + _prime = ", i_prime + j_prime)

        return i, j, ij

    @staticmethod
    def compute_overall_center_of_gravity(polygons):
        """Computes the overall center of gravity of a list of polygons."""
        total_area = 0.0
        weighted_cx_sum = 0.0
        weighted_cy_sum = 0.0

        for poly in polygons:
            area = poly[AREA]
            barycenter_x, barycenter_y = poly[BARYCENTER]

            total_area += area
            weighted_cx_sum += area * barycenter_x
            weighted_cy_sum += area * barycenter_y

        if total_area == 0:
            return 0.0, 0.0

        x_g = weighted_cx_sum / total_area
        y_g = weighted_cy_sum / total_area
        print("total_area", total_area)
        return x_g, y_g

    @staticmethod
    def compute_combined_absolute_moment_of_inertia(polygons):
        """Computes the combined absolute moments of inertia of a list of polygons."""
        jx_total, jy_total, jxy_total = 0.0, 0.0, 0.0

        for poly in polygons:
            jxi, jyi, jxyi = poly[ABSOLUTE_MOMENTS_OF_INERTIA]
            jx_total += jxi
            jy_total += jyi
            jxy_total += jxyi

        return jx_total, jy_total, jxy_total

    @staticmethod
    def compute_combined_baricentric_moments_of_inertia(polygons, x_G, y_G):
        """Computes the combined principal baricentric moments of inertia (i, j)

        of a list of polygons at the system center of gravity (x_G, y_G).
        """
        jx_total, jy_total, jxy_total = 0.0, 0.0, 0.0
        area_total = 0.0

        for poly in polygons:
            jxi, jyi, jxyi = poly[ABSOLUTE_MOMENTS_OF_INERTIA]
            area_total += poly[AREA]
            jx_total += jxi
            jy_total += jyi
            jxy_total += jxyi

        # 1. Trasporto d'Inerzia Huygens-Steiner al baricentro globale del sistema
        jx = jx_total - y_G**2 * area_total
        jy = jy_total - x_G**2 * area_total
        jxy = jxy_total - y_G * x_G * area_total

        # 2. Calcolo dei momenti principali d'inerzia (min, max)
        r = math.sqrt(((jx - jy) / 2.0) ** 2 + jxy**2)
        i = (jx + jy) / 2.0 - r
        j = (jx + jy) / 2.0 + r
        
        print("Baricenter: ", x_G, y_G)

        return i, j