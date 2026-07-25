from muse_toolbox.utils import sample_parameter
import random
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
from shapely.geometry import Point
import trimesh


class MicrophoneArray:
    """
    Generates microphone array coordinates based on abstract geometric rules.

    This class creates a blueprint for an array, centered at the origin,
    defined by its geometry, distribution type, number of mics, and a
    single sizing parameter: the maximum distance between any two mics.
    """

    def __init__(
        self,
        num_mics: int,
        geometry: str,
        distribution: str,
        max_distance: float | None = None,  # Made optional
        min_distance: float = 0.0,
        **kwargs,  # Added to capture 'radius'
    ):
        """
        Constructs the array blueprint and generates the local coordinates.

        Args:
            num_mics: The number of microphones.
            geometry: The shape to distribute mics in. One of:
                      ['line', 'circle', 'disk', 'square_area',
                       'sphere_surface', 'sphere_volume', 'cube_volume'].
            distribution: How to place mics. One of: ['regular', 'random'].
            max_distance: The maximum distance between any two mics in the array.
            min_distance: The minimum required distance between any two mics.
        """
        self.num_mics = sample_parameter(num_mics)
        self.geometry = geometry
        self.distribution = distribution
        # Handle 'radius' for double_tetraeder or custom configs
        self.radius = None
        if "radius" in kwargs:
            self.radius = sample_parameter(kwargs["radius"])
        self.max_distance = (
            max_distance if max_distance is not None else self._infer_max_distance()
        )
        self.min_distance = min_distance

        # Determine if the geometry is fundamentally 2D or 3D
        self._2d_geometries = ["line", "circle", "disk", "square_area"]
        self.dimensionality = 2 if self.geometry in self._2d_geometries else 3

        # This holds the final coordinates, shape (3, num_mics), centered at origin.
        self.local_locations = self._generate_locations()
        self.distance_matrix = self._compute_distance_matrix()

    def _infer_max_distance(self) -> float:
        """
        Infers a reasonable max_distance based on the geometry and number of mics.
        This is a fallback if max_distance is not provided.

        Returns:
            A float representing the inferred maximum distance between any two mics.
        """
        if self.geometry == "double_tetraeder":
            return self.radius * 2 if self.radius is not None else 1.0
        elif self.geometry == "circle":
            if self.radius is None:
                raise NotImplementedError("radius must be provided for circular geometry")
            return self.radius * 2
        raise NotImplementedError(
            "Automatic max_distance inference is not implemented yet. "
            "Please provide max_distance explicitly when creating a MicrophoneArray instance."
        )
        # if self.geometry == "line":
        #     return self.num_mics - 1  # Assuming unit spacing for regular line
        # elif self.geometry == "circle":
        #     return self.num_mics / np.pi  # Approximate circumference for regular circle
        # elif self.geometry == "disk":
        #     return np.sqrt(self.num_mics)  # Approximate diameter for regular disk
        # elif self.geometry == "square_area":
        #     return np.sqrt(
        #         2 * self.num_mics
        #     )  # Diagonal of a square with num_mics points
        # elif self.geometry in ["sphere_surface", "sphere_volume"]:
        #     return (
        #         self.num_mics ** (1 / 3)
        #     ) * 2  # Approximate diameter for regular sphere
        # elif self.geometry == "cube_volume":
        #     return (
        #         self.num_mics ** (1 / 3)
        #     ) * 2  # Approximate diagonal for regular cube
        # elif self.geometry == "double_tetraeder":
        #     return self.radius * 2 if self.radius is not None else 1.0
        # else:
        #     raise NotImplementedError(f"Geometry '{self.geometry}' is not recognized.")

    @classmethod
    def from_locations(cls, local_locations: np.ndarray):
        """
        Alternative constructor to create a MicrophoneArray from pre-defined locations.
        This method will automatically center the provided locations and determine
        if the geometry is 2D (co-planar) or 3D.

        Args:
            local_locations (np.ndarray): A numpy array of shape (3, num_mics)
                                          containing the microphone coordinates.
        """
        # Validate input shape
        if (
            not isinstance(local_locations, np.ndarray)
            or local_locations.ndim != 2
            or local_locations.shape[0] != 3
        ):
            raise ValueError(
                f"local_locations must be a numpy array of shape (3, N), but got shape {local_locations.shape}"
            )

        num_mics = local_locations.shape[1]

        # 1. Center the locations by subtracting their center of mass.
        if num_mics > 0:
            center_of_mass = np.mean(local_locations, axis=1, keepdims=True)
            centered_locations = local_locations - center_of_mass
        else:
            centered_locations = local_locations

        # 2. Determine dimensionality by checking for co-planarity.
        # If 3 or fewer points, they are always co-planar.
        if num_mics <= 3:
            dimensionality = 2
        else:
            # Use SVD to find the variance along principal axes.
            # If the smallest singular value is near zero, the points are co-planar.
            _u, s, _vh = np.linalg.svd(centered_locations.T)
            is_planar = np.isclose(s[-1], 0)
            dimensionality = 2 if is_planar else 3

        # 3. Calculate max and min distances from the centered points
        if num_mics > 1:
            from scipy.spatial.distance import pdist

            pairwise_distances = pdist(centered_locations.T)
            max_dist = float(np.max(pairwise_distances))
            min_dist = float(np.min(pairwise_distances))
        else:
            max_dist = 0.0
            min_dist = 0.0

        # 4. Instantiate the class with inferred/placeholder params
        instance = cls(
            num_mics=num_mics,
            geometry="custom",
            distribution="custom",
            max_distance=max_dist,
            min_distance=min_dist,
        )

        # 5. Override locations and dimensionality
        instance.local_locations = centered_locations
        instance.dimensionality = dimensionality

        return instance

    def _generate_locations(self) -> np.ndarray:
        """
        Dispatcher method that calls the correct helper based on geometry and distribution.
        """
        # If geometry is 'custom', it means locations are provided externally.
        if self.geometry == "custom":
            return np.zeros((3, self.num_mics))  # Return a placeholder

        # A mapping from (geometry, distribution) to the appropriate helper function.
        generation_methods = {
            ("line", "regular"): self._generate_regular_line,
            ("line", "random"): self._generate_random_line,
            ("circle", "regular"): self._generate_regular_circle,
            ("circle", "random"): self._generate_random_circle,
            ("disk", "regular"): self._generate_regular_disk,
            ("disk", "random"): self._generate_random_disk,
            ("sphere_surface", "regular"): self._generate_regular_sphere_surface,
            ("sphere_surface", "random"): self._generate_random_sphere_surface,
            ("sphere_volume", "regular"): self._generate_regular_sphere_volume,
            ("sphere_volume", "random"): self._generate_random_sphere_volume,
            ("cube_volume", "random"): self._generate_random_cube_volume,
            ("square_area", "random"): self._generate_random_square_area,
            ("gen2D", "random"): self._generate_random_2D_array_old,
            ("gen3D", "random"): self._generate_random_3D_array_old,
            ("double_tetraeder", "fixed"): self._generate_regular_double_tetraeder,
        }

        method_key = (self.geometry, self.distribution)
        if method_key not in generation_methods:
            raise NotImplementedError(
                f"The combination of geometry='{self.geometry}' and "
                f"distribution='{self.distribution}' is not supported."
            )

        # Call the selected helper function.
        return generation_methods[method_key]()

    def _compute_distance_matrix(self) -> np.ndarray:
        """Computes the pairwise distance matrix between microphones."""
        if self.num_mics == 0:
            return np.zeros((0, 0))

        # Using broadcasting to compute pairwise distances efficiently.
        diff = self.local_locations[:, :, None] - self.local_locations[:, None, :]
        dist_matrix = np.linalg.norm(diff, axis=0)

        return dist_matrix

    # --- Helper Methods ---

    def _generate_regular_line(self) -> np.ndarray:
        """Generates regularly spaced points on a line of length `max_distance`."""
        if self.num_mics == 0:
            return np.zeros((3, 0))

        if self.num_mics == 1:
            # A single microphone is always at the center.
            return np.zeros((3, 1))

        # Use np.linspace to create evenly spaced points from -half to +half length.
        # This ensures the total distance between the first and last mic is `max_distance`.
        half_length = self.max_distance / 2.0
        x_coords = np.linspace(-half_length, half_length, num=self.num_mics)

        # Create the final 3D coordinate array, placing points along the x-axis.
        locations = np.zeros((3, self.num_mics))
        locations[0, :] = x_coords

        return locations

    def _generate_random_line(self) -> np.ndarray:
        """Generates uniformly random points on a line of length `max_distance`."""
        if self.num_mics == 0:
            return np.zeros((3, 0))

        if self.num_mics == 1:
            # A single microphone is always at the center.
            return np.zeros((3, 1))

        half_length = self.max_distance / 2.0

        # Sample `num_mics` points from a uniform distribution
        # within the bounds [-half_length, half_length].
        x_coords = np.random.uniform(-half_length, half_length, size=self.num_mics)

        # Create the final 3D coordinate array, placing points along the x-axis.
        locations = np.zeros((3, self.num_mics))
        locations[0, :] = x_coords

        return locations

    def _generate_regular_circle(self) -> np.ndarray:
        """Generates regularly spaced points on a circle's circumference."""
        if self.num_mics == 0:
            return np.zeros((3, 0))

        if self.num_mics == 1:
            # A single microphone is always at the center.
            return np.zeros((3, 1))

        radius = self.max_distance / 2.0

        # Generate evenly spaced angles from 0 to 2*pi.
        # endpoint=False is important to avoid duplicating the first point at 2*pi.
        angles = np.linspace(0, 2 * np.pi, num=self.num_mics, endpoint=False)

        # Convert polar coordinates (radius, angle) to Cartesian (x, y).
        x_coords = radius * np.cos(angles)
        y_coords = radius * np.sin(angles)

        # Create the final 3D coordinate array in the xy-plane.
        locations = np.zeros((3, self.num_mics))
        locations[0, :] = x_coords
        locations[1, :] = y_coords

        return locations

    def _generate_random_circle(self) -> np.ndarray:
        """Generates uniformly random points on a circle's circumference."""
        if self.num_mics == 0:
            return np.zeros((3, 0))

        if self.num_mics == 1:
            # A single microphone is always at the center.
            return np.zeros((3, 1))

        radius = self.max_distance / 2.0

        # Generate random angles from a uniform distribution between 0 and 2*pi.
        angles = np.random.uniform(0, 2 * np.pi, size=self.num_mics)

        # Convert polar coordinates (radius, angle) to Cartesian (x, y).
        x_coords = radius * np.cos(angles)
        y_coords = radius * np.sin(angles)

        # Create the final 3D coordinate array in the xy-plane.
        locations = np.zeros((3, self.num_mics))
        locations[0, :] = x_coords
        locations[1, :] = y_coords

        return locations

    def _generate_regular_disk(self) -> np.ndarray:
        """
        Generates regularly spaced points within a 2D disk using a Fermat's spiral pattern.
        This ensures a quasi-uniform distribution for any number of microphones.
        """
        if self.num_mics == 0:
            return np.zeros((3, 0))

        if self.num_mics == 1:
            # A single microphone is always at the center.
            return np.zeros((3, 1))

        radius = self.max_distance / 2.0

        # Golden angle for spiral distribution
        golden_angle = np.pi * (3.0 - np.sqrt(5.0))

        # Create an index for each point (0 to N-1)
        indices = np.arange(self.num_mics)

        # Calculate the radius for each point. The sqrt ensures uniform area distribution.
        # The radius scales up to the maximum radius for the last point.
        radii = radius * np.sqrt(indices / (self.num_mics - 1))

        # Calculate the angle for each point using the golden angle
        angles = golden_angle * indices

        # Convert polar coordinates to Cartesian
        x_coords = radii * np.cos(angles)
        y_coords = radii * np.sin(angles)

        # Create the final 3D coordinate array in the xy-plane.
        locations = np.zeros((3, self.num_mics))
        locations[0, :] = x_coords
        locations[1, :] = y_coords

        return locations

    def _generate_random_disk(self) -> np.ndarray:
        """Generates uniformly random points within a 2D disk."""
        if self.num_mics == 0:
            return np.zeros((3, 0))

        if self.num_mics == 1:
            # A single microphone is always at the center.
            return np.zeros((3, 1))

        radius = self.max_distance / 2.0

        # Generate random angles uniformly
        angles = np.random.uniform(0, 2 * np.pi, size=self.num_mics)

        # Generate random radii. Taking the sqrt of a uniform variable
        # ensures that the points are uniformly distributed by area.
        sqrt_radii = radius * np.sqrt(np.random.uniform(0, 1, size=self.num_mics))

        # Convert polar coordinates to Cartesian
        x_coords = sqrt_radii * np.cos(angles)
        y_coords = sqrt_radii * np.sin(angles)

        # Create the final 3D coordinate array in the xy-plane.
        locations = np.zeros((3, self.num_mics))
        locations[0, :] = x_coords
        locations[1, :] = y_coords

        return locations

    def _generate_regular_sphere_surface(self) -> np.ndarray:
        """
        Generates quasi-regularly spaced points on a sphere's surface using a Fibonacci lattice.
        """
        if self.num_mics == 0:
            return np.zeros((3, 0))

        # For a single mic on a surface, place it at an arbitrary point on the surface, e.g., along the x-axis.
        if self.num_mics == 1:
            locations = np.zeros((3, 1))
            locations[0, 0] = self.max_distance / 2.0
            return locations

        radius = self.max_distance / 2.0

        # Golden angle for spiral distribution on a sphere
        golden_angle = np.pi * (3.0 - np.sqrt(5.0))

        # Create an index for each point (0 to N-1)
        indices = np.arange(self.num_mics)

        # Calculate the y-coordinate (latitude). This distributes points evenly along the y-axis.
        y = 1 - (2 * indices) / (self.num_mics - 1)

        # Calculate the radius of the circle at that y-height
        radius_at_y = np.sqrt(1 - y**2)

        # Calculate the angle (longitude) for each point using the golden angle
        theta = golden_angle * indices

        # Convert to Cartesian coordinates (unit sphere)
        x = radius_at_y * np.cos(theta)
        z = radius_at_y * np.sin(theta)

        # Scale by the desired radius and combine into the final array
        locations = np.vstack([x, y, z]) * radius

        return locations

    def _generate_random_sphere_surface(self) -> np.ndarray:
        """Generates uniformly random points on a sphere's surface."""
        if self.num_mics == 0:
            return np.zeros((3, 0))

        # For a single mic on a surface, place it at an arbitrary point on the surface, e.g., along the x-axis.
        if self.num_mics == 1:
            locations = np.zeros((3, 1))
            locations[0, 0] = self.max_distance / 2.0
            return locations

        radius = self.max_distance / 2.0

        # Generate points from a 3D Gaussian distribution.
        # When normalized, these points are uniformly distributed on a sphere's surface.
        locations = np.random.randn(3, self.num_mics)

        # Normalize each column (each point) to have a length of 1.
        norms = np.linalg.norm(locations, axis=0)
        locations /= norms

        # Scale the points to the desired radius.
        locations *= radius

        return locations

    def _generate_regular_sphere_volume(self) -> np.ndarray:
        """
        Generates quasi-regularly spaced points within a 3D sphere using a 3D Fibonacci lattice.
        """
        print(
            "WARNING: This approximate method for regular distribution in a sphere volume has still a major error."
        )
        if self.num_mics == 0:
            return np.zeros((3, 0))

        if self.num_mics == 1:
            # A single microphone is always at the center.
            return np.zeros((3, 1))

        radius = self.max_distance / 2.0

        # Create an index for each point (0 to N-1)
        indices = np.arange(self.num_mics) - self.num_mics / 2.0
        # Shift by 0.5 for better distribution
        # if num_mics is even then add 0.5 to center points better
        if self.num_mics % 2 == 0:
            indices += 0.5
        else:
            indices += 1.0

        indices_alt = []
        for ind in indices:
            if ind > 0:
                indices_alt.append(ind)
                indices_alt.append(-ind)
            elif ind == 0:
                indices_alt.append(ind)
            else:
                continue
        indices_alt = np.array(indices_alt)  # [: self.num_mics])

        # --- 1. Calculate the radial distance for each point ---
        # The cubic root ensures uniform volume distribution.
        # We use (indices + 0.5) to avoid a point at the exact center (radius=0)
        # and to distribute points more evenly.
        radii = radius * np.cbrt((indices_alt) / self.num_mics)

        # --- 2. Calculate the angular components using the golden angle ---
        # This distributes points evenly on a sphere's surface.
        golden_angle = np.pi * (1 + np.sqrt(5))  # Use the other golden ratio variant

        # Azimuthal angle (phi)
        phi = golden_angle * indices

        # Polar angle (theta)
        theta = np.pi * np.cos(2 * indices / self.num_mics)
        # cos_theta = 1 - (2 * indices + 1) / self.num_mics
        # sin_theta = np.sqrt(1 - cos_theta**2)

        # --- 3. Convert spherical to Cartesian coordinates ---
        # Each point's direction is determined by the angles, and its distance
        # from the origin is determined by its corresponding radius.
        x = radii * np.sin(theta) * np.cos(phi)
        y = radii * np.sin(theta) * np.sin(phi)
        z = radii * np.cos(theta)

        locations = np.vstack([x, y, z])

        return locations

    def _generate_random_sphere_volume(self) -> np.ndarray:
        """Generates uniformly random points within a 3D sphere."""
        if self.num_mics == 0:
            return np.zeros((3, 0))

        if self.num_mics == 1:
            # A single microphone is always at the center.
            return np.zeros((3, 1))

        radius = self.max_distance / 2.0

        # 1. Generate random directions uniformly on a sphere surface.
        # This is done by sampling from a 3D Gaussian and normalizing.
        locations = np.random.randn(3, self.num_mics)
        norms = np.linalg.norm(locations, axis=0)
        # Avoid division by zero for the unlikely case of a zero vector
        norms[norms == 0] = 1
        locations /= norms

        # 2. Generate random radii. To ensure uniform volume distribution,
        # the radii must be sampled from a distribution whose PDF is
        # proportional to r^2. This is achieved by taking the cube root
        # of a uniform random variable.
        random_radii = radius * np.cbrt(np.random.uniform(0, 1, size=self.num_mics))

        # 3. Scale the unit direction vectors by the random radii.
        locations *= random_radii

        return locations

    def _generate_random_square_area(self) -> np.ndarray:
        """
        Generates uniformly random points within a 2D square, respecting a minimum distance.
        """
        if self.num_mics == 0:
            return np.zeros((3, 0))

        if self.num_mics == 1:
            # A single microphone is always at the center.
            return np.zeros((3, 1))

        # --- 1. Pre-check for physical impossibility (2D version) ---
        # Based on the densest packing of equal circles in a plane (hexagonal packing),
        # which can fill about 90.69% of the total area.
        if self.min_distance > 0:
            # The area of the exclusion circle around each microphone.
            area_of_exclusion_circle = np.pi * (self.min_distance / 2) ** 2
            total_exclusion_area = self.num_mics * area_of_exclusion_circle

            # The total area of the square in which mics are placed.
            square_area = self.max_distance**2

            # Maximum theoretical packing density for circles in 2D.
            max_packing_density = np.pi / (2 * np.sqrt(3))  # Approx. 0.9069
            max_fillable_area = square_area * max_packing_density

            assert total_exclusion_area <= max_fillable_area, (
                f"Configuration is physically impossible due to circle packing limits. "
                f"Required exclusion area for {self.num_mics} mics ({total_exclusion_area:.4f} m^2) "
                f"exceeds the maximum fillable area of the square ({max_fillable_area:.4f} m^2), "
                f"which is ~90.7% of the total square area ({square_area:.4f} m^2)."
            )

        # --- 2. Iterative placement ("dart throwing") ---
        half_length = self.max_distance / 2.0
        locations = np.zeros((3, self.num_mics))
        max_attempts_per_mic = 1000  # Failsafe to prevent infinite loops

        for i in range(self.num_mics):
            for attempt in range(max_attempts_per_mic):
                # Generate a random candidate point in the XY plane
                candidate_xy = np.random.uniform(-half_length, half_length, size=2)
                candidate = np.array([candidate_xy[0], candidate_xy[1], 0.0]).reshape(
                    3, 1
                )

                # If it's the first point or min_distance is zero, accept it immediately
                if i == 0 or self.min_distance == 0:
                    locations[:, i] = candidate.flatten()
                    break

                # Check distance to all previously placed points
                distances = np.linalg.norm(locations[:, :i] - candidate, axis=0)
                if np.all(distances >= self.min_distance):
                    locations[:, i] = candidate.flatten()
                    break  # Valid point found, move to the next mic
            else:
                # This 'else' belongs to the 'for attempt' loop.
                raise RuntimeError(
                    f"Failed to place microphone #{i+1} after {max_attempts_per_mic} "
                    f"attempts. The configuration with num_mics={self.num_mics}, "
                    f"max_distance={self.max_distance}, and min_distance={self.min_distance} "
                    f"is likely too dense to solve."
                )

        return locations

    def _generate_random_cube_volume(self) -> np.ndarray:
        """
        Generates uniformly random points within a 3D cube, respecting a minimum distance.
        """
        if self.num_mics == 0:
            return np.zeros((3, 0))

        if self.num_mics == 1:
            # A single microphone is always at the center.
            return np.zeros((3, 1))

        # --- 1. Pre-check for physical impossibility ---
        # Based on the Kepler conjecture, the densest possible packing of equal spheres
        # in 3D space can only fill about 74% of the total volume. We use this to
        # provide a much stricter and more realistic check for dense configurations.
        if self.min_distance > 0:
            # The volume of the exclusion sphere around each microphone.
            # The radius of this sphere is half the minimum distance.
            volume_of_exclusion_sphere = (4 / 3) * np.pi * (self.min_distance / 2) ** 3
            total_exclusion_volume = self.num_mics * volume_of_exclusion_sphere

            # The total volume of the cube in which mics are placed.
            cube_volume = self.max_distance**3

            # Maximum theoretical packing density for spheres in 3D.
            max_packing_density = np.pi / (3 * np.sqrt(2))  # Approx. 0.74048
            max_fillable_volume = cube_volume * max_packing_density

            assert total_exclusion_volume <= max_fillable_volume, (
                f"Configuration is physically impossible due to sphere packing limits. "
                f"Required exclusion volume for {self.num_mics} mics ({total_exclusion_volume:.4f} m^3) "
                f"exceeds the maximum fillable volume of the cube ({max_fillable_volume:.4f} m^3), "
                f"which is ~74% of the total cube volume ({cube_volume:.4f} m^3)."
            )

        # --- 2. Iterative placement ("dart throwing") ---
        half_length = self.max_distance / 2.0
        locations = np.zeros((3, self.num_mics))
        max_attempts_per_mic = 1000  # Failsafe to prevent infinite loops

        for i in range(self.num_mics):
            for attempt in range(max_attempts_per_mic):
                # Generate a random candidate point
                candidate = np.random.uniform(
                    -half_length, half_length, size=3
                ).reshape(3, 1)

                # If it's the first point or min_distance is zero, accept it immediately
                if i == 0 or self.min_distance == 0:
                    locations[:, i] = candidate.flatten()
                    break

                # Check distance to all previously placed points
                # `locations[:, :i]` slices the already placed microphones
                distances = np.linalg.norm(locations[:, :i] - candidate, axis=0)
                if np.all(distances >= self.min_distance):
                    locations[:, i] = candidate.flatten()
                    break  # Valid point found, move to the next mic
            else:
                # This 'else' belongs to the 'for attempt' loop.
                # It runs only if the loop completes without a 'break'.
                raise RuntimeError(
                    f"Failed to place microphone #{i+1} after {max_attempts_per_mic} "
                    f"attempts. The configuration with num_mics={self.num_mics}, "
                    f"max_distance={self.max_distance}, and min_distance={self.min_distance} "
                    f"is likely too dense to solve, even if theoretically possible."
                )

        return locations

    def _generate_random_2D_array_old(self) -> np.ndarray:
        """
        Generates uniformly random points in a 2D plane, constrained only by min/max
        inter-microphone distances.

        The process is as follows:
        1. Iteratively place microphones ("dart throwing") ensuring only the `min_distance`
           is respected. The initial placement area is unbounded.
        2. After all points are placed, find the maximum pairwise distance in the generated set.
        3. Scale the entire array down so that this maximum distance equals `self.max_distance`.
        4. Center the final array by subtracting its center of mass.
        """
        if self.num_mics == 0:
            return np.zeros((3, 0))

        if self.num_mics == 1:
            return np.zeros((3, 1))

        locations = [np.array([0.0, 0.0, 0.0])]
        max_attempts_per_mic = 100

        # 1. Iteratively place points respecting min_distance
        for i in range(1, self.num_mics):
            for attempt in range(max_attempts_per_mic):

                # # Pick a random existing point to place the new point near
                # anchor_point = locations[:, np.random.randint(i)]

                # Calculate center of mass of current points
                current_mics = np.column_stack(locations)
                center_of_mass = np.mean(current_mics, axis=1, keepdims=True)

                # Find the index of the mic closest to the center of mass
                distances_to_com = np.linalg.norm(current_mics - center_of_mass, axis=0)
                closest_mic_index = np.argmin(distances_to_com)

                # Set the anchor point to this closest mic
                anchor_point = current_mics[:, closest_mic_index]

                # Generate a candidate point in a random direction, at least min_distance away
                r = np.sqrt(
                    np.random.uniform(self.min_distance**2, self.max_distance**2)
                )
                angle = np.random.uniform(0, 2 * np.pi)
                offset = np.array([r * np.cos(angle), r * np.sin(angle), 0.0])
                candidate = anchor_point.reshape(3, 1) + offset.reshape(3, 1)

                # Check distance to all previously placed points
                distances = np.linalg.norm(current_mics[:, :i] - candidate, axis=0)
                if np.all(
                    (self.max_distance >= distances) & (distances >= self.min_distance)
                ):
                    print(
                        f"Placed mic #{i+1} at {candidate.flatten()} after {attempt+1} attempts."
                    )
                    locations.append(candidate.flatten())
                    break  # Valid point found
            else:
                # raise RuntimeError(
                #     f"Failed to place microphone #{i+1} after {max_attempts_per_mic} "
                #     f"attempts. The min_distance={self.min_distance} might be too large."
                # )
                print(
                    f"Failed to place microphone #{i+1} after {max_attempts_per_mic} "
                    f"attempts. The min_distance={self.min_distance} might be too large."
                )
                break

        # # 2. Scale the array to enforce max_distance
        # if self.num_mics > 1:
        #     from scipy.spatial.distance import pdist

        #     pairwise_distances = pdist(locations.T)
        #     current_max_dist = np.max(pairwise_distances)

        #     if current_max_dist > 0:
        #         scale_factor = self.max_distance / current_max_dist
        #         locations *= scale_factor

        # 3. Center the array
        center_of_mass = np.mean(np.column_stack(locations), axis=1, keepdims=True)
        centered_locations = np.column_stack(locations) - center_of_mass

        return centered_locations

    def _generate_random_3D_array_old(self) -> np.ndarray:
        """
        Generates uniformly random points in 3D space using a "place-then-scale" method.

        The process is as follows:
        1. Iteratively place microphones ("dart throwing") ensuring each new point
           is within [min_distance, max_distance] of all existing points.
        2. After all points are placed, find the maximum pairwise distance in the set.
        3. Scale the entire array down so that this maximum distance equals `self.max_distance`.
        4. Center the final array by subtracting its center of mass.
        """
        if self.num_mics == 0:
            return np.zeros((3, 0))

        if self.num_mics == 1:
            return np.zeros((3, 1))

        locations = [np.array([0.0, 0.0, 0.0])]
        max_attempts_per_mic = 100

        # 1. Iteratively place points
        for i in range(1, self.num_mics):
            for attempt in range(max_attempts_per_mic):

                # # Pick a random existing point to place the new point near
                # anchor_point = locations[:, np.random.randint(i)]

                # Calculate center of mass of current points
                current_mics = np.column_stack(locations)
                center_of_mass = np.mean(current_mics, axis=1, keepdims=True)

                # Find the index of the mic closest to the center of mass
                distances_to_com = np.linalg.norm(current_mics - center_of_mass, axis=0)
                closest_mic_index = np.argmin(distances_to_com)

                # Set the anchor point to this closest mic
                anchor_point = current_mics[:, closest_mic_index]

                # Generate a candidate point in a random 3D direction
                r = np.cbrt(
                    np.random.uniform(self.min_distance**3, self.max_distance**3)
                )
                # Generate a random 3D unit vector for the direction
                direction = np.random.randn(3)
                direction /= np.linalg.norm(direction)
                offset = r * direction
                candidate = anchor_point.reshape(3, 1) + offset.reshape(3, 1)

                # Check distance to all previously placed points
                distances = np.linalg.norm(current_mics - candidate, axis=0)
                if np.all(
                    (self.max_distance >= distances) & (distances >= self.min_distance)
                ):
                    print(
                        f"Placed mic #{i+1} at {candidate.flatten()} after {attempt+1} attempts."
                    )
                    locations.append(candidate.flatten())
                    break  # Valid point found
            else:
                # raise RuntimeError(
                #     f"Failed to place microphone #{i+1} after {max_attempts_per_mic} "
                #     f"attempts. The constraints might be too tight."
                # )
                print(
                    f"Failed to place microphone #{i+1} after {max_attempts_per_mic} "
                    f"attempts. The constraints might be too tight."
                )
                break

        # # 2. Scale the array to enforce the global max_distance
        # if self.num_mics > 1:
        #     from scipy.spatial.distance import pdist

        #     pairwise_distances = pdist(locations.T)
        #     current_max_dist = np.max(pairwise_distances)

        #     if current_max_dist > 0:
        #         scale_factor = self.max_distance / current_max_dist
        #         locations *= scale_factor

        # 3. Center the array
        center_of_mass = np.mean(np.column_stack(locations), axis=1, keepdims=True)
        centered_locations = np.column_stack(locations) - center_of_mass

        return centered_locations

    def _generate_random_2D_array(self) -> np.ndarray:
        """
        Generates random points in a 2D plane where each new point is constrained
        to be within a min/max distance from ALL existing points.

        The process uses the 'shapely' library:
        1. Start with one microphone at the origin.
        2. For each subsequent microphone:
           a. Calculate the valid placement area. This is the geometric intersection
              of annuli (rings) defined by the min/max distance from every
              existing microphone.
           b. Sample a random point from within this valid area.
        3. After all points are placed, center the final array by subtracting its
           center of mass so the array's centroid is at the origin.
        """

        if self.num_mics == 0:
            return np.zeros((3, 0))

        if self.num_mics == 1:
            return np.zeros((3, 1))

        # Start with the first mic at the origin
        mic_locations = [np.array([0.0, 0.0, 0.0])]
        center_point = Point(mic_locations[0][0], mic_locations[0][1])
        max_circle = center_point.buffer(self.max_distance)
        min_circle = center_point.buffer(self.min_distance)
        valid_area = max_circle.difference(min_circle)

        max_sampling_attempts = 10000  # Per point

        # Iteratively place the remaining microphones
        for i in range(1, self.num_mics):

            # --- 2. Check if a valid placement is possible ---
            if valid_area.is_empty:
                # raise RuntimeError(
                #     f"Failed to place microphone #{i+1}: No valid placement area exists. "
                #     f"The constraints (min_dist={self.min_distance}, max_dist={self.max_distance}) "
                #     f"are likely too tight for {self.num_mics} microphones."
                # )
                print(
                    f"Failed to place microphone #{i+1}: No valid placement area exists. "
                    f"The constraints (min_dist={self.min_distance}, max_dist={self.max_distance}) "
                    f"are likely too tight for {self.num_mics} microphones."
                )
                break

            # --- 3. Sample a point from within the valid area ---
            min_x, min_y, max_x, max_y = valid_area.bounds
            new_mic_point = None
            for j in range(max_sampling_attempts):
                candidate = Point(
                    np.random.uniform(min_x, max_x), np.random.uniform(min_y, max_y)
                )
                if valid_area.contains(candidate):
                    print(f"Placed mic #{i+1} at {candidate} after {j+1} attempts.")
                    new_mic_point = candidate
                    break

            if new_mic_point is None:
                # raise RuntimeError(
                #     f"Failed to sample a point for microphone #{i+1} after "
                #     f"{max_sampling_attempts} attempts. The valid area might be "
                #     f"too small or fragmented."
                # )
                print(
                    f"Failed to sample a point for microphone #{i+1} after "
                    f"{max_sampling_attempts} attempts. The valid area might be "
                    f"too small or fragmented."
                )
                break

            # --- 4. Add the new microphone to the list ---
            new_mic_location = np.array([new_mic_point.x, new_mic_point.y, 0.0])
            mic_locations.append(new_mic_location)

            newPoint = Point(new_mic_location[0], new_mic_location[1])
            max_circle = newPoint.buffer(self.max_distance)
            min_circle = newPoint.buffer(self.min_distance)
            annulus = max_circle.difference(min_circle)
            valid_area = valid_area.intersection(annulus)

        # --- 5. Finalize the array ---
        # Convert list of arrays to a single (N, 3) numpy array
        final_locations = np.array(mic_locations)

        # Center the array by subtracting its center of mass
        center_of_mass = np.mean(final_locations, axis=0, keepdims=True)
        centered_locations = final_locations - center_of_mass

        # Return in the required (3, N) shape
        return centered_locations.T

    def _generate_random_3D_array(self) -> np.ndarray:
        """
        Generates random points in 3D space where each new point is constrained
        to be within a min/max distance from ALL existing points.

        The process uses the 'trimesh' library:
        1. Start with one microphone at the origin.
        2. For each subsequent microphone:
           a. Calculate the valid placement volume. This is the geometric
              intersection of spherical shells defined by the min/max distance
              from every existing microphone.
           b. Sample a random point from within this valid volume.
        3. After all points are placed, center the final array by subtracting its
           center of mass so the array's centroid is at the origin.
        """

        if self.num_mics == 0:
            return np.zeros((3, 0))

        if self.num_mics == 1:
            return np.zeros((3, 1))

        # Start with the first mic at the origin
        mic_locations = [np.array([0.0, 0.0, 0.0])]
        max_sampling_attempts = 10000  # Per point

        max_sphere = trimesh.primitives.Sphere(
            radius=self.max_distance, center=mic_locations[0]
        )
        min_sphere = trimesh.primitives.Sphere(
            radius=self.min_distance, center=mic_locations[0]
        )
        valid_volume = max_sphere.difference(min_sphere)

        # Iteratively place the remaining microphones
        for i in range(1, self.num_mics):

            # --- 2. Check if a valid placement is possible ---
            if valid_volume.is_empty:
                # raise RuntimeError(
                #     f"Failed to place microphone #{i+1}: No valid placement volume exists. "
                #     f"The constraints (min_dist={self.min_distance}, max_dist={self.max_distance}) "
                #     f"are likely too tight for {self.num_mics} microphones."
                # )
                print(
                    f"Failed to place microphone #{i+1}: No valid placement volume exists. "
                    f"The constraints (min_dist={self.min_distance}, max_dist={self.max_distance}) "
                    f"are likely too tight for {self.num_mics} microphones."
                )
                break

            # --- 3. Sample a point from within the valid volume ---
            min_b, max_b = valid_volume.bounds
            new_mic_point = None
            for j in range(max_sampling_attempts):
                candidate = np.random.uniform(low=min_b, high=max_b)
                if valid_volume.contains([candidate]):
                    print(f"Placed mic #{i+1} at {candidate} after {j+1} attempts.")
                    new_mic_point = candidate
                    break

            if new_mic_point is None:
                # raise RuntimeError(
                #     f"Failed to sample a point for microphone #{i+1} after "
                #     f"{max_sampling_attempts} attempts. The valid volume might be "
                #     f"too small or fragmented."
                # )
                print(
                    f"Failed to sample a point for microphone #{i+1} after "
                    f"{max_sampling_attempts} attempts. The valid volume might be "
                    f"too small or fragmented."
                )
                break

            # --- 4. Add the new microphone to the list ---
            mic_locations.append(new_mic_point)
            max_sphere = trimesh.primitives.Sphere(
                radius=self.max_distance, center=new_mic_point
            )
            min_sphere = trimesh.primitives.Sphere(
                radius=self.min_distance, center=new_mic_point
            )
            shell = max_sphere.difference(min_sphere)
            valid_volume = valid_volume.intersection(shell)

        # --- 5. Finalize the array ---
        # Convert list of arrays to a single (N, 3) numpy array
        final_locations = np.array(mic_locations)

        # Center the array by subtracting its center of mass
        center_of_mass = np.mean(final_locations, axis=0, keepdims=True)
        centered_locations = final_locations - center_of_mass

        # Return in the required (3, N) shape
        return centered_locations.T

    def _generate_regular_double_tetraeder(self) -> np.ndarray:
        """
        Generates a fixed double tetrahedron (Stella Octangula configuration).
        Consists of a Normal Tetrahedron (Tele) and an Inverted Tetrahedron (T2).
        Each has its center of mass at the origin.
        """
        # Ensure we have a valid radius
        if self.radius is None:
            self.radius = self.max_distance / 2.0

        R = self.radius

        # --- Tetra 1 (Normal) ---
        # Upright: One face pointing down (-Z), opposite corner pointing up (+Z).
        # Top vertex: (0, 0, R)
        # Base plane: z = -R/3
        # Radius of base circle: r_base = sqrt(R^2 - (R/3)^2) = 2*sqrt(2)/3 * R
        r_base = (2 * np.sqrt(2) / 3) * R
        h_base = -R / 3.0

        # Orientation:
        # "One edge parallel to first dimension (X)"
        # "A corner opposite of pointing towards positive second dimension" -> Pointing to -Y.

        # Base Vertex 1 (Pointing -Y):
        t1_v1 = [0, r_base, h_base]
        # Base Vertex 2 (210 deg):
        t1_v2 = [
            r_base * np.cos(np.deg2rad(210)),
            r_base * np.sin(np.deg2rad(210)),
            h_base,
        ]
        # Base Vertex 3 (330 deg):
        t1_v3 = [
            r_base * np.cos(np.deg2rad(330)),
            r_base * np.sin(np.deg2rad(330)),
            h_base,
        ]
        # Top Vertex:
        t1_top = [0, 0, R]

        # --- Tetra 2 (Inverted) ---
        # "Corners above the centerpoints of the faces of the normal one".
        # This describes the dual tetrahedron, which is strictly T2 = -T1.
        # Alternatively, defined by reflection or rotation logic provided:
        # Bottom vertex: (0, 0, -R)
        # Base plane: z = +R/3 (Since it's inverted)

        # Inverted coordinates are just negative of Normal coordinates
        t2_v1 = [-x for x in t1_v1]  # Points to +Y
        t2_v2 = [-x for x in t1_v2]
        t2_v3 = [-x for x in t1_v3]
        t2_bottom = [-x for x in t1_top]  # (0, 0, -R)

        # Collect all points. Order: [T1_Top, T1_Base1, T1_Base2, T1_Base3, T2_Bottom, T2_Base1, T2_Base2, T2_Base3]
        # You can adjust order if specific channel mapping is needed.
        locations = np.array(
            [t1_top, t1_v1, t1_v2, t1_v3, t2_bottom, t2_v1, t2_v2, t2_v3]
        ).T

        # Check num_mics
        if self.num_mics != 8:
            print(
                f"Warning: 'double_tetraeder' geometry generates 8 mics, but {self.num_mics} were requested. Using 8."
            )

        return locations

    def place(
        self,
        room_dims: list[float],
        min_dist_from_walls: float | None,  # CHANGE: Allow None
        plot_filepath: str | Path | None = None,
        restrict_rot_2_xy_plane: bool = False,
        fix_height: float | None = None,
        fixed_position: list[float] | None = None,  # CHANGE: New argument
        fixed_rotation: bool = False,  # CHANGE: New argument
    ) -> np.ndarray:
        """
        Places the array in a room with a random center and orientation.

        If the array is 2D and `restrict_rot_2_xy_plane` was set to True, the
        orientation is restricted to the XY-plane. Otherwise, a full 3D
        rotation is applied.
        """

        # 0. Warn if 2D rotation is requested for a 3D geometry and override.
        if restrict_rot_2_xy_plane and self.dimensionality == 3:
            warnings.warn(
                f"Restricted rotation to 2D for a 3D geometry ('{self.geometry}'). "
                f"Is this really desired?"
            )

        # 1. Determine Center
        if fixed_position is not None:
            center = np.array(fixed_position)
            # Optional: Check if center is in room
            if np.any(center < 0) or np.any(center > np.array(room_dims)):
                raise ValueError(
                    f"Fixed position {fixed_position} is outside room {room_dims}"
                )
        else:
            # Random placement logic
            safe_dist_walls = (
                min_dist_from_walls if min_dist_from_walls is not None else 0.0
            )
            effective_radius = self.max_distance / 2.0
            min_dist = safe_dist_walls + effective_radius

            center_x = random.uniform(min_dist, room_dims[0] - min_dist)
            center_y = random.uniform(min_dist, room_dims[1] - min_dist)
            if fix_height is not None:
                center_z = fix_height
            else:
                center_z = random.uniform(min_dist, room_dims[2] - min_dist)
            center = np.array([center_x, center_y, center_z])

        # 2. Apply a random rotation based on the flag
        if fixed_rotation:
            rotation_matrix = np.eye(3)
        elif restrict_rot_2_xy_plane:
            # Apply a 2D rotation around the Z-axis
            angle = np.random.uniform(0, 2 * np.pi)
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            rotation_matrix = np.array(
                [[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]]
            )
        else:
            # Apply a full random 3D rotation
            from scipy.spatial.transform import Rotation as R

            rotation_matrix = R.random().as_matrix()

        rotated_locs = rotation_matrix @ self.local_locations

        # 3. Translate to the final center position in the room
        global_locs = rotated_locs + center[:, np.newaxis]

        # 4. Plot the global locations if a filepath is provided
        if plot_filepath:
            self.plot_global_locations(global_locs, room_dims, plot_filepath)

        return global_locs

    # --- Plotting Methods ---

    def plot_local_locations(self, filepath: str | Path = "Playground/MA_local.png"):
        """
        Generates and saves a 2x2 plot visualizing the local (blueprint) coordinates.
        """
        fig, axs = plt.subplots(2, 2, figsize=(10, 10))
        fig.suptitle("Microphone Array Blueprint (Local Coordinates)", fontsize=16)
        plot_margin = self.max_distance * 0.6

        # 3D Plot (Top-Left)
        ax_3d = fig.add_subplot(2, 2, 1, projection="3d")
        ax_3d.scatter(
            xs=self.local_locations[0, :],
            ys=self.local_locations[1, :],
            zs=self.local_locations[2, :],  # type: ignore
            c="r",
            marker="o",
        )
        ax_3d.set_title("3D View")
        ax_3d.set_xlabel("X")
        ax_3d.set_ylabel("Y")
        ax_3d.set_zlabel("Z")
        ax_3d.set_xlim([-plot_margin, plot_margin])
        ax_3d.set_ylim([-plot_margin, plot_margin])
        ax_3d.set_zlim([-plot_margin, plot_margin])
        ax_3d.set_box_aspect([1, 1, 1])
        ax_3d.grid(True)

        # XY Projection (Top-Right)
        ax_xy = axs[0, 1]
        ax_xy.scatter(
            self.local_locations[0, :], self.local_locations[1, :], c="r", marker="o"
        )
        ax_xy.set_title("XY Projection (Top-Down View)")
        ax_xy.set_xlabel("X")
        ax_xy.set_ylabel("Y")
        ax_xy.set_xlim([-plot_margin, plot_margin])
        ax_xy.set_ylim([-plot_margin, plot_margin])
        ax_xy.set_aspect("equal", adjustable="box")
        ax_xy.grid(True)

        # XZ Projection (Bottom-Left)
        ax_xz = axs[1, 0]
        ax_xz.scatter(
            self.local_locations[0, :], self.local_locations[2, :], c="r", marker="o"
        )
        ax_xz.set_title("XZ Projection (Front View)")
        ax_xz.set_xlabel("X")
        ax_xz.set_ylabel("Z")
        ax_xz.set_xlim([-plot_margin, plot_margin])
        ax_xz.set_ylim([-plot_margin, plot_margin])
        ax_xz.set_aspect("equal", adjustable="box")
        ax_xz.grid(True)

        # YZ Projection (Bottom-Right)
        ax_yz = axs[1, 1]
        ax_yz.scatter(
            self.local_locations[1, :], self.local_locations[2, :], c="r", marker="o"
        )
        ax_yz.set_title("YZ Projection (Side View)")
        ax_yz.set_xlabel("Y")
        ax_yz.set_ylabel("Z")
        ax_yz.set_xlim([-plot_margin, plot_margin])
        ax_yz.set_ylim([-plot_margin, plot_margin])
        ax_yz.set_aspect("equal", adjustable="box")
        ax_yz.grid(True)

        fig.tight_layout(rect=(0, 0, 1, 0.96))
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(filepath)
        plt.close(fig)
        print(f"Saved local locations plot to: {filepath}")

    @staticmethod
    def plot_global_locations(
        global_locs: np.ndarray,
        room_dims: list[float],
        filepath: str | Path = "Playground/MA_in_room.png",
    ):
        """
        Generates and saves a 2x2 plot visualizing the global (placed) coordinates.
        """
        fig, axs = plt.subplots(2, 2, figsize=(10, 10))
        fig.suptitle(
            "Microphone Array Placed in Room (Global Coordinates)", fontsize=16
        )

        # 3D Plot (Top-Left)
        ax_3d = fig.add_subplot(2, 2, 1, projection="3d")
        ax_3d.scatter(
            xs=global_locs[0, :],
            ys=global_locs[1, :],
            zs=global_locs[2, :],  # type: ignore
            c="b",
            marker="^",
        )
        ax_3d.set_title("3D View")
        ax_3d.set_xlabel("X")
        ax_3d.set_ylabel("Y")
        ax_3d.set_zlabel("Z")
        ax_3d.set_xlim([0, room_dims[0]])
        ax_3d.set_ylim([0, room_dims[1]])
        ax_3d.set_zlim([0, room_dims[2]])
        ax_3d.set_box_aspect(
            (
                np.ptp(ax_3d.get_xlim()),
                np.ptp(ax_3d.get_ylim()),
                np.ptp(ax_3d.get_zlim()),
            )
        )
        ax_3d.grid(True)

        # XY Projection (Top-Right)
        ax_xy = axs[0, 1]
        ax_xy.scatter(global_locs[0, :], global_locs[1, :], c="b", marker="^")
        ax_xy.set_title("XY Projection")
        ax_xy.set_xlabel("X")
        ax_xy.set_ylabel("Y")
        ax_xy.set_xlim([0, room_dims[0]])
        ax_xy.set_ylim([0, room_dims[1]])
        ax_xy.set_aspect("equal", adjustable="box")
        ax_xy.grid(True)

        # XZ Projection (Bottom-Left)
        ax_xz = axs[1, 0]
        ax_xz.scatter(global_locs[0, :], global_locs[2, :], c="b", marker="^")
        ax_xz.set_title("XZ Projection")
        ax_xz.set_xlabel("X")
        ax_xz.set_ylabel("Z")
        ax_xz.set_xlim([0, room_dims[0]])
        ax_xz.set_ylim([0, room_dims[2]])
        ax_xz.set_aspect("equal", adjustable="box")
        ax_xz.grid(True)

        # YZ Projection (Bottom-Right)
        ax_yz = axs[1, 1]
        ax_yz.scatter(global_locs[1, :], global_locs[2, :], c="b", marker="^")
        ax_yz.set_title("YZ Projection")
        ax_yz.set_xlabel("Y")
        ax_yz.set_ylabel("Z")
        ax_yz.set_xlim([0, room_dims[1]])
        ax_yz.set_ylim([0, room_dims[2]])
        ax_yz.set_aspect("equal", adjustable="box")
        ax_yz.grid(True)

        fig.tight_layout(rect=(0, 0, 1, 0.96))
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(filepath)
        plt.close(fig)
        print(f"Saved global locations plot to: {filepath}")
