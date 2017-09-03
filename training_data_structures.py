from typing import List, Set, Dict

import word_analysis as ana

class TrainingSetElement:

    def __init__(self, word_a: str, word_b: str) -> None:
        self.__word_a = word_a
        self.__word_b = word_b
        lcs_matrix = ana.LCSMatrix(word_a, word_b)
        subsequence_intervals = ana.WordSubsequenceIntervals(lcs_matrix)
        transformation = ana.build_word_transformation(subsequence_intervals)
        self.__subsequence_intervals = subsequence_intervals
        self.__transformation = transformation

    @property
    def word_a(self) -> str:
        return self.__word_a

    @property
    def word_b(self) -> str:
        return self.__word_b

    @property
    def transformation(self) -> ana.WordTransformation:
        return self.__transformation

    @property
    def subsequence_intervals(self) -> ana.WordSubsequenceIntervals:
        return self.__subsequence_intervals

    def __hash__(self) -> int:
        return hash(self.transformation)

    def __repr__(self) -> str:
        return "({}, {}, {})".format(self.word_a, self.word_b, repr(self.transformation))


class Cluster:

    def __init__(self, first_item: TrainingSetElement):
        self.__transformation = first_item.transformation # type: ana.WordTransformation
        self.__items = {first_item} # type: Set[TrainingSetElement]

    def can_add_item(self, item: TrainingSetElement) -> bool:
        if self.__transformation.maybe_joinable(item.transformation):
            joined_transformation = self.__transformation.join(item.transformation)
            for e in self.__items:
                if joined_transformation.apply(e.word_a) != e.word_b:
                    return False
            if joined_transformation.apply(item.word_a) != item.word_b:
                return False
            return True
        return False

    def add_item(self, item: TrainingSetElement) -> bool:
        if self.can_add_item(item):
            joined_transformation = self.__transformation.join(item.transformation)
            self.__transformation = joined_transformation
            self.__items.add(item)
            return True
        return False

    @property
    def transformation(self) -> ana.WordTransformation:
        return self.__transformation

    @property
    def items(self) -> Set[TrainingSetElement]:
        return self.__items.copy()

    def __repr__(self) -> str:
        return "<Cluster, {}, [{}] {} elements>".format(str(self.transformation), self.__items, len(self.__items))


class FrozenCluster():

    def __init__(self, cluster: Cluster) -> None:
        self.__cluster = cluster

    @property
    def transformation(self) -> ana.WordTransformation:
        return self.__cluster.transformation

    @property
    def items(self) -> Set[TrainingSetElement]:
        return self.__cluster.items

    def __repr__(self) -> str:
        return repr(self.__cluster)


class ClusterSet:

    def __init__(self) -> None:
        self.__clusters = dict() # type: Dict[int, List[Cluster]]

    def add(self, elem: TrainingSetElement) -> None:
        key = hash(elem)
        if key not in self.__clusters:
            self.__clusters[key] = [Cluster(elem)]
        else:
            for cluster in self.__clusters[key]:
                if cluster.add_item(elem):
                    return
            self.__clusters[key].append(Cluster(elem))

    def get_clusters(self) -> List[FrozenCluster]:
        result = []
        for _, clusterset in self.__clusters.items():
            for cluster in clusterset:
                result.append(FrozenCluster(cluster))
        return result