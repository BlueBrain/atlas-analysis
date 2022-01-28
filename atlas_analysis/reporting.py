""" Hierarchy of classes producing reports on
annotation files and other altas related volumetric files
"""
from collections import defaultdict
import codecs
import json

import numpy as np
from scipy import ndimage
from scipy.ndimage.morphology import generate_binary_structure, binary_fill_holes

import voxcell
from atlas_analysis.exceptions import AtlasAnalysisError


class Report:
    """Abstract class generating a report based on a VoxelData or its raw numpy array.

    A report is created after inspection of an annotation nrrd file or, more generally,
    after inspection of one or more atlas-related volumetric files such as
    density or orientation files.

    Report fields are stored in dedicated member variables.
    Reports can be saved under the form of a json file and can be loaded from a json file.
    Report objects can be turned into Python dicts through the to_dict() method.
    """

    @classmethod
    def load(cls, file_name):
        """Build a report out of a json file.

        Args:
            file_name(str): the name of the json file to be loaded.

        Returns:
            Report object.
        """
        with codecs.open(file_name, 'r', encoding='utf-8') as report_file:
            dictionary = json.load(report_file)
            return cls.from_dict(dictionary)

    @classmethod
    def from_dict(cls, dictionary):
        """Build a report out of a Python dict.

        Args:
            dictionary(dict): the dictionary used for field extraction.

        Returns:
            Report object.

        Raises:
            AtlasAnalysisError for each missing key.
        """
        raise NotImplementedError()

    def to_dict(self):
        """Build a Python dict out of the report member fields.

        Returns:
            Python dict where each report field is instantiated as a key
            with a corresponding field value.
        """
        raise NotImplementedError()

    def save_as(self, file_name):
        """Save the report under the form of a json file.

        Args:
            file_name(str): the name of the json file to be created.
        """
        with codecs.open(file_name, 'w', encoding='utf-8') as report_file:
            json.dump(self.to_dict(), report_file, separators=(',', ':'), indent=4)


class VoxelDataReport(Report):
    """Abstract class to generate a report built out of a VoxelData object.
    """

    @classmethod
    def from_voxel_data(cls, voxel_data, **kwargs):
        """Build a report out of a VoxelData object.

        Args:
            voxel_data(VoxelData): the VoxelData object subject to reporting.

        Returns:
            Report object.
        """
        raise NotImplementedError()


class RawReport(VoxelDataReport):
    """Abstract class to generate a report built out of a VoxelData object or a raw numpy array.

    This report class is used when all the information required by the report can be extracted
    from the raw numpy array only.
    """

    @classmethod
    def from_raw(cls, raw, **kwargs):
        """Build a report out of a numpy array.

        Args:
            raw(numpy.ndarray): the numpy array subject to reporting.

        Returns:
            Report object.
        """
        raise NotImplementedError()

    @classmethod
    def from_voxel_data(cls, voxel_data, **kwargs):
        """Build a report out of a VoxelData object.

        Args:
            voxel_data(VoxelData): the VoxelData object subject to reporting.

        Returns:
            Report object.
        """
        return cls.from_raw(voxel_data.raw, **kwargs)


class DensityReport(Report):
    """Class to generate density reports from density nrrd files.

    A report is created after inspection of an annotation nrrd file and a list of
    density nrrd files. The report is restricted to a list of specified leaf identifiers,
    i.e., a list of integers representing leaf regions in the annotation hierarchy.
    """

    def __init__(self, dictionary):
        """Args:
            dictionary(dict): dict whose keys are strings representing integers
                and whose values are dict of the form
            {
                'cell_density': 0.001,
                'glia_density': 0.003,
                'exc_density': 0.004,
                ...
            }
            Each key corresponds to the integer identifier of a leaf region.

            Attributes:
                dictionary(dict): the input dictionary.
        """
        self.dictionary = dictionary

    @classmethod
    def from_dict(cls, dictionary):
        return cls(dictionary)

    def to_dict(self):
        """
        Ouput example:
            '1': {
                'cell_density': 0.001,
                'glia_density': 0.003,
                'exc_density': 0.004
            },
            '2': {
                'cell_density': 0.001,
                'glia_density': 0.001,
                'exc_density': 0.001
            },
            '3': {
                'cell_density': 0.0007,
                'glia_density': 0.002,
                'exc_density': 0.0007
            }
        """
        return self.dictionary

    @classmethod
    def from_files(cls, annotation_voxel_data, filepaths, leaf_ids):
        """Generate a report containing the average densities of the specified leaf regions.

        Densities are expressed in (um)^{-3}

        Args:
            voxel_data(VoxelData): the VoxelData object subject to density reporting.
            filepapths(list): list of density file paths (Path objects), with names such as
                cell_density.nrrd, exc_density.nrrd, inh_density.nrrd, glia_density.nrrd,
                ...
            leaf_ids(list): integer list of unique identifiers of the leaf regions.

        Note: the voxel_data object should hold a 3D integer array consisting of the labels
        of all the regions covered by the density files. This array should have the same shape
        as the 3D float arrays contained in the density files. All VoxelData objects are
        assumed to have the same voxel dimensions.

        Returns:
            DensityReport object: the returned instance holds the dict self.dictionary
                whose keys are the leaf_ids of the input list and whose values
                are dicts of the form {'cell_density': 0.0001, 'glia_density': 0.0002, ...}
                where each float value is the average density of the
                corresponding region for the correponding density type.
                The leaf id keys are integers converted to strings.
                Densities are expressed in (um)^{-3}.

        """
        density_dictionary = defaultdict(dict)
        raw = annotation_voxel_data.raw
        voxel_volume = annotation_voxel_data.voxel_volume

        for filepath in filepaths:
            density_type = filepath.name.replace('.nrrd', '')
            density_voxel_data = voxcell.VoxelData.load_nrrd(filepath)
            density_raw = density_voxel_data.raw
            for leaf_id in leaf_ids:
                leaf_id_mask = raw == leaf_id
                density = np.mean(density_raw[leaf_id_mask]) / voxel_volume
                # Densities are expressed in (um)^{-3}
                density_dictionary[str(leaf_id)][density_type] = density

        return cls(dict(density_dictionary))


class Histogram:
    """Class holding the data structure of an histogram with integer bin edges.

    The class offers the to_dict() and from_dict() methods used
    in the context of report writing.
    """

    def __init__(self, histogram_values, bin_edges, total):
        """Args:
            histogram_values(array-like): 1D int array
                holding the count of each bin
            bin_edges(array-like): 1D int array holding the bin edges.
                A bin is a pair of consecutive edges of the form
                [lower, upper) where lower is excluded and upper is excluded.
                We have len(bin_edges) = len(histogram_values) + 1
            total(int): the total count of elements used to produce the histogram.
                This count may be greater than the sum of all bin counts as bins may only
                cover a strict subset of these elements.

            Attributes:
                histogram_values(array-like): 1D int array holds the histgram values.
                bin_edges(array-like): 1D int array holding the bin edges.
                total(int): total count of elements used to produce the histogram.
        """
        self.histogram_values = histogram_values
        self.bin_edges = bin_edges
        if len(self.bin_edges) != len(self.histogram_values) + 1:
            raise AtlasAnalysisError(
                'The number of bins doesn\'t match the number of histogram values.'
            )
        if total < sum(histogram_values):
            raise AtlasAnalysisError(
                'The total count is less then than the sum of all bin counts.'
            )
        self.total = total

    def to_dict(self):
        """Build a Python dict out of the data members.

        Returns:
            Python dict with the upper edges of the histogram bins as keys
            and the bin counts as values.
            The histogram zero-values corresponding to the bins whose upper bound
            is larger than the total count are not displayed.
            The latter dictionary has an extra key, namely,
            'total', which corresponds to the number of elements used to
            create the histogram. This count may be greater than the sum of all bin counts
            Ouput example:
            {
                'total': 1455,
                '0': 0
                '10': 101,
                '100': 552,
                '1000': 674,
                '10000': 15,
                '100000': 1,
                '1000000': 0,
                '10000000': 0,
                '100000000': 0
            }
            In the above example:
                101 elements lie within the bounds [0, 10).
                552 elements lie within the bounds [10, 100).
        """
        dictionary = {}
        dictionary['total'] = self.total
        # display the least bin edge: there is no element in the empty range (0, 0]
        dictionary[str(self.bin_edges[0])] = 0
        for i in range(len(self.bin_edges) - 1):
            # display count only for sensible bin edges
            dictionary[str(self.bin_edges[i + 1])] = int(self.histogram_values[i])
        return dictionary

    @classmethod
    def from_dict(cls, dictionary):
        """Build an Histogram instance out of a Python dict.

        Args:
            dictionary(dict): the dictionary used for field extraction.

        Returns:
            Histogram object.

        Raises:
            AtlasAnalysisError: If the 'total' key is missing,
            if there is a non-integer key different from 'total',
            or if there are less than two bin edges, i.e., integer keys.
        """
        total = None
        bin_edges = []
        error_message = 'Invalid input dictionary: '
        for key, value in dictionary.items():
            if key == 'total':
                total = value
            else:
                edge = None
                try:
                    edge = int(key)
                except ValueError as error:
                    raise AtlasAnalysisError(
                        error_message + f'Unknown non-integer key {key}. {error}'
                    ) from error
                bin_edges.append(edge)

        # Few more validity checks
        if total is None:
            raise AtlasAnalysisError(error_message + 'missing \'total\' key.')
        if len(bin_edges) < 2:
            raise AtlasAnalysisError(
                error_message
                + 'at least two integer keys, i.e., two bin edges are required.'
            )
        bin_edges.sort()
        min_edge_bin = bin_edges[0]
        histogram_values = [
            dictionary[str(bin_edge)]
            for bin_edge in bin_edges
            if bin_edge != min_edge_bin
        ]
        return cls(histogram_values, bin_edges, total)


class ConnectivityReport(RawReport):
    """Report connnectivity information for the input annotated atlas or region.

    The data members indicate whether the input array is connected or not.
    The instance reports the total number of connected components distinct
    from the background (the background has zero value) and
    stores an histogram of the connected components based on their sizes.
    The size of a connected component is its number of voxels.
    The bins of the histogram are of the form [lower_size, upper_size) = [10**i, 10**(i + 1))
    with i <= 7 where lower_size is included and upper_size is excluded.

    Ouput example of the to_dict() method:
    {
        'is_connected': False,
        'connected_components_histogram':
            'total': 1455,
            '0': 0
            '10': 101,
            '100': 552,
            '1000': 674,
            '10000': 15,
            '100000': 1,
            '1000000': 0,
            '10000000': 0,
            '100000000': 0
    }
    The report indicates that the input annotated atlas or region is disconnected.
    It has 1'455 connected components. The largest connected component has less than
    100'000 voxels but at least 10'000 voxels.
    """

    BINS = (0, 10, 10 ** 2, 10 ** 3, 10 ** 4, 10 ** 5, 10 ** 6, 10 ** 7, 10 ** 8)

    def __init__(self, connected_component_histogram):
        """
        Args:
            connected_component_histogram(Histogram): histogram of
                the connected componenents based on their sizes.
                The size of a connected component is its number of voxels.
                The bins of the histogram are of the form
                [lower_size, upper_size) = [10**i, 10**(i + 1)) with i <= 7
                and where lower_size is included and upper_size is excluded.

        Attributes:
            connected_component_histogram(Histogram): stores the histogram
                of connected components.
        """
        self.connected_component_histogram = connected_component_histogram

    @classmethod
    def from_raw(cls, raw, **kwargs):
        structure = generate_binary_structure(3, 1)
        labeled_components, number_of_components = ndimage.label(
            raw, structure=structure
        )
        unique_values, counts = np.unique(labeled_components, return_counts=True)
        # Remove the connected component corresponding to the background
        counts = counts[unique_values > 0]
        assert len(counts) == number_of_components
        histogram_values, bin_edges = np.histogram(counts, bins=cls.BINS)
        connected_component_histogram = Histogram(
            histogram_values, bin_edges, len(counts)
        )
        return cls(connected_component_histogram)

    def to_dict(self):
        dictionary = {}
        dictionary['is_connected'] = self.connected_component_histogram.total == 1
        dictionary[
            'connected_component_histogram'
        ] = self.connected_component_histogram.to_dict()
        return dictionary

    @classmethod
    def from_dict(cls, dictionary):
        for key in ['is_connected', 'connected_component_histogram']:
            if key not in dictionary:
                raise AtlasAnalysisError(
                    f'Invalid input dictionary. Missing key {key}.'
                )

        connected_component_histogram = Histogram.from_dict(
            dictionary['connected_component_histogram']
        )
        number_of_components = connected_component_histogram.total
        if dictionary['is_connected'] and number_of_components > 2:
            raise AtlasAnalysisError(
                f'Inconsistent input dictionary. is_connected is True '
                f'whereas the number of connected components is {number_of_components}.'
            )
        return cls(connected_component_histogram)


class CavityReport(RawReport):
    """Report cavity information for the input annotated atlas or region.

    A report instance holds the count of cavity voxels and their
    proportion with respect to the non-zero voxels.
    A cavity is a hole nested in a thick part of a volume.

    Ouput example of the to_dict() method:
    {
        'has_cavities': True,
        'voxel_count': 2500,
        'proportion': 0.005
    }
    The report indicates that the region of interest (possibly a complete annotated atlas)
    has some cavities. These cavities consist of 2'500 voxels, which amounts to 0.5% of
    the overall volume.
    """

    def __init__(self, cavity_voxel_count, proportion):
        """
        Args:
            cavity_voxel_count(int): number of cavity voxels.
            proportion(float): proportion of cavity voxels
                with respect to the non-zero voxels.

        Attributes:
            voxel_count(int): number of cavity voxels.
            proportion(float): proportion of of cavity voxels
                with respect to the non-zero voxels.
        """
        self.voxel_count = cavity_voxel_count
        self.proportion = proportion

    @classmethod
    def from_raw(cls, raw, **kwargs):
        mask = raw != 0
        structure = generate_binary_structure(3, 3)
        filled_mask = binary_fill_holes(mask, structure=structure)
        cavities = filled_mask != mask
        voxel_count = np.count_nonzero(cavities)
        return cls(voxel_count, float(voxel_count) / np.count_nonzero(mask))

    def to_dict(self):
        dictionary = {}
        dictionary['has_cavities'] = bool(self.voxel_count > 0)
        dictionary['voxel_count'] = self.voxel_count
        dictionary['proportion'] = self.proportion
        return dictionary

    @classmethod
    def from_dict(cls, dictionary):
        for key in ['has_cavities', 'voxel_count', 'proportion']:
            if key not in dictionary:
                raise AtlasAnalysisError(
                    f'Invalid input dictionary. Missing key {key}.'
                )

        if not dictionary['has_cavities'] and dictionary['voxel_count'] > 0:
            raise AtlasAnalysisError(
                f"Inconsistent input dictionary. has_cavities is False "
                f"whereas the number of cavity voxels is {dictionary['voxel_count']}."
            )
        return cls(dictionary['voxel_count'], dictionary['proportion'])


class HeaderReport(VoxelDataReport):
    """Report class encompassing the nrrd header of a VoxelData object.

    Ouput example of the to_dict() method:
    {
        'sizes': [234, 545, 657],
        'space_dimension': 3,
        'space_directions': [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]
        'space_origin': [230.0, 250.0, 100.0]
    }
    """

    def __init__(self, sizes, space_dimension, space_directions, space_origin):
        """
        Args:
            sizes(list): list of integers of length space_dimension
            space_dimension(int): dimension of the VoxelData object,
                usually 2 (pixels) or 3 (voxels).
            space_directions(list): list of float lists, where each float list represents
                a vector of dimension space_dimensions. The vector length is the voxel side length
                along this vector. There are space_dimension pairwise orthogonal vectors.
            space_origin(list): list of space_dimension floats representing the origin offset
                with respect to the reference frame.

        Attributes:
            sizes(list): input sizes.
            space_dimension(int): input space_dimension.
            space_directions(list): input space_directions.
            space_origin(list): input space_origin
        """
        self.sizes = sizes
        self.space_dimension = space_dimension
        self.space_directions = space_directions
        self.space_origin = space_origin

    @classmethod
    def from_voxel_data(cls, voxel_data, **kwargs):
        return cls(
            list(voxel_data.raw.shape),
            voxel_data.ndim,
            # The conversion of numpy arrays to lists is needed
            # when serializing to JSON.
            np.diag(voxel_data.voxel_dimensions).tolist(),
            voxel_data.offset.tolist(),
        )

    def to_dict(self):
        dictionary = {
            'sizes': self.sizes,
            'space_dimension': self.space_dimension,
            'space_directions': self.space_directions,
            'space_origin': self.space_origin,
        }
        return dictionary

    @classmethod
    def from_dict(cls, dictionary):
        for key in ['sizes', 'space_dimension', 'space_directions', 'space_origin']:
            if key not in dictionary:
                raise AtlasAnalysisError(
                    f'Invalid input dictionary. Missing key {key}.'
                )

        return cls(
            dictionary['sizes'],
            dictionary['space_dimension'],
            dictionary['space_directions'],
            dictionary['space_origin'],
        )


class RegionVoxelCountReport(RawReport):
    """Class reporting voxel counts of a leaf region.

    In addition to the region voxel count and the leaf region proportion wrt to the
    overall annotation file, this class is composed of two optional
    region-specific reports:
        - connected components count (connectivity)
        - cavities voxel count (cavities)
    """

    def __init__(self, voxel_count, proportion, connectivity=None, cavities=None):
        """
        Args:
            voxel_count(int): region voxel count
            proportion(float): proportion of the leaf region voxels wrt
                the number of non-zero voxels of the annotated nrrd file from which
                the region has been extracted.
            connectivity(ConnectivityReport): (optional) report containing connectivity
                information on the leaf region. It indicates in particular the number of connected
                components. Defaults to None.
            cavities(CavityReport): (optional) report on the cavities of the leaf region.
                It indicates in particular the count of cavity voxels.
                Defaults to None.
        Attributes:
            voxel_count(int): input voxel_count.
            proportion(float): input proportion.
            connectivity(ConnectivityReport): input connectivity.
            cavities(CavityReport): input cavities.
        """
        self.voxel_count = voxel_count
        self.proportion = proportion
        self.connectivity = connectivity
        self.cavities = cavities

    @classmethod
    def from_raw(cls, raw, **kwargs):
        voxel_count = kwargs['voxel_count']
        proportion = kwargs['proportion']
        identifier = kwargs['identifier']
        connectivity = None
        cavities = None

        if (
            'connectivity_is_required' in kwargs
            and kwargs['connectivity_is_required'] is True
        ):
            connectivity = ConnectivityReport.from_raw(raw == int(identifier))
        if (
            'cavities_are_required' in kwargs
            and kwargs['cavities_are_required'] is True
        ):
            cavities = CavityReport.from_raw(raw == int(identifier))

        return cls(voxel_count, proportion, connectivity, cavities)

    def to_dict(self):
        dictionary = {
            'voxel_count': self.voxel_count,
            'proportion': self.proportion,
        }
        if self.connectivity is not None:
            dictionary['connectivity'] = self.connectivity.to_dict()

        if self.cavities is not None:
            dictionary['cavities'] = self.cavities.to_dict()

        return dictionary

    @classmethod
    def from_dict(cls, dictionary):
        for key in ['voxel_count', 'proportion']:
            if key not in dictionary:
                raise AtlasAnalysisError(
                    f'Invalid input dictionary. Missing key {key}.'
                )
        connectivity = None
        cavities = None
        if 'connectivity' in dictionary:
            connectivity = ConnectivityReport.from_dict(dictionary['connectivity'])
        if 'cavities' in dictionary:
            cavities = CavityReport.from_dict(dictionary['cavities'])

        return cls(
            dictionary['voxel_count'], dictionary['proportion'], connectivity, cavities
        )


class VoxelCountReport(VoxelDataReport):
    """Report voxel counts for the whole VoxelData object and for its individual leaf regions.

    This class is a composition of several reports both at
    the global level (annotation file) and the local level (leaf region):
        - voxel count
        - connected components count (connectivity)
        - cavities voxel count (cavities)

    In addition, it appends the header information of the corresponding annotation nrrd file.

    For an output example of the to_dict() method, see tests/test_reporting.py.
    """
    # pylint: disable=too-many-instance-attributes
    def __init__(
        self,
        header,
        voxel_count,
        non_zero_voxel_count,
        region_list,
        region_counts,
        connectivity=None,
        cavities=None,
        region_map=None,
    ):
        """
        Args:
            header(HeaderReport): HeaderReport instance holding the VoxelData header information.
            voxel_count(int): voxel count of the whole VoxelData object.
            non_zero_voxel_count(int): voxel count of non-zero voxels (the brackground is removed).
            region_list(list): list of leaf region identifiers.
                Identifiers are strings representing integers.
                Strings were preferred to integers because the json output format
                only supports string keys.
            region_counts(dict): dict whose keys are leaf region identifiers (str)
                and whose values are RegionVoxelCountReport instances.
            connectivity(ConnectivityReport): (optional) report on the connectivity
                of the VoxelData object.
                It indicates in particular the number of connected components
                of the VoxelData object. Defaults to None.
            cavities(CavitiyReport): (optional) report on the cavities of the VoxelData object.
                It indicates in particular the count of cavity voxels.
                A cavity is a hole nested in a thick part of a volume.
                Defaults to None.
            region_map(voxcell.RegionMap): an object to navigate the brain regions hierarchy.

        Attributes:
            voxel_count(int): input voxel_count
            non_zero_voxel_count(int): non_zero_voxel_count
            region_list(list): input region_list
            connectivity(ConnectivityReport): input connectivity
            cavities(CavitiyReport): input cavities
            region_counts(dict): input region_counts
            region_map(voxcell.RegionMap): input region map
        """
        # pylint: disable=too-many-instance-attributes
        self.header = header
        self.voxel_count = voxel_count
        self.non_zero_voxel_count = non_zero_voxel_count
        self.region_list = region_list
        self.region_counts = region_counts
        self.connectivity = connectivity
        self.cavities = cavities
        self.region_map = region_map

    @classmethod
    def from_voxel_data(cls, voxel_data, **kwargs):
        # Global information
        raw = voxel_data.raw
        non_zero_voxel_count = np.count_nonzero(raw)
        region_ids, counts = np.unique(raw, return_counts=True)
        non_zero_indices = np.nonzero(region_ids)  # Removes the background
        region_ids = region_ids[non_zero_indices]
        counts = counts[non_zero_indices]
        region_list = region_ids.tolist()
        region_list = [str(identifier) for identifier in region_ids]
        connectivity = None
        cavities = None
        if kwargs.get('connectivity_is_required', False):
            connectivity = ConnectivityReport.from_raw(raw)
        if kwargs.get('cavities_are_required', False):
            cavities = CavityReport.from_raw(raw)
        # Region specific information
        region_counts = {}
        for identifier, count in zip(region_list, counts):
            region_kwargs = {
                'identifier': identifier,
                # The next two values are not re-computed
                # from raw for each region. They are directly passed to
                # RegionVoxelCountReport.from_raw() instead.
                # This is done for performance reasons.
                'voxel_count': int(count),
                'proportion': float(count) / non_zero_voxel_count,
            }
            region_kwargs.update(kwargs)
            region_counts[identifier] = RegionVoxelCountReport.from_raw(
                raw, **region_kwargs
            )

        return cls(
            HeaderReport.from_voxel_data(voxel_data),
            int(np.prod(raw.shape)),
            non_zero_voxel_count,
            region_list,
            region_counts,
            connectivity,
            cavities,
            region_map=kwargs.get('region_map', None),
        )

    def to_dict(self):
        region_counts_dict = {}
        for identifier, report in self.region_counts.items():
            region_counts_dict[identifier] = report.to_dict()

        dictionary = {
            'header': self.header.to_dict(),
            'voxel_count': self.voxel_count,
            'non_zero_voxel_count': self.non_zero_voxel_count,
            'proportion': float(self.non_zero_voxel_count) / self.voxel_count,
            'number_of_regions': len(self.region_list),
            'region_list': self.region_list,
            'region_counts': region_counts_dict,
        }
        if self.region_map is not None:
            dictionary['region_map'] = {
                id_: {
                    'acronym': self.region_map.get(int(id_), 'acronym'),
                    'name': self.region_map.get(int(id_), 'name'),
                }
                for id_ in self.region_list
            }
        if self.connectivity is not None:
            dictionary['connectivity'] = self.connectivity.to_dict()
        if self.cavities is not None:
            dictionary['cavities'] = self.cavities.to_dict()
        return dictionary

    @classmethod
    def from_dict(cls, dictionary):
        for key in [
            'header',
            'voxel_count',
            'non_zero_voxel_count',
            'proportion',
            'number_of_regions',
            'region_list',
            'region_counts',
        ]:
            if key not in dictionary:
                raise AtlasAnalysisError(
                    f'Invalid input dictionary. Missing key {key}.'
                )
        region_counts = {}
        for identifier, report_dict in dictionary['region_counts'].items():
            region_counts[identifier] = RegionVoxelCountReport.from_dict(report_dict)
        connectivity = None
        cavities = None
        if dictionary.get('connectivity', False):
            connectivity = ConnectivityReport.from_dict(dictionary['connectivity'])
        if dictionary.get('cavities', False):
            cavities = CavityReport.from_dict(dictionary['cavities'])

        return cls(
            HeaderReport.from_dict(dictionary['header']),
            dictionary['voxel_count'],
            dictionary['non_zero_voxel_count'],
            dictionary['region_list'],
            region_counts,
            connectivity,
            cavities,
        )
