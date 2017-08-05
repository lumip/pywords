from typing import TypeVar, List, FrozenSet, Callable

T = TypeVar('T')


class ClusterSet:

    def __init__(self) -> None:
        self.__clusters = dict()

    def add(self, elem: T) -> None:
        key = hash(elem)
        if key not in self.__clusters:
            self.__clusters[key] = [{elem}]
        else:
            for cluster in self.__clusters[key]:
                if next(iter(cluster)) == elem:
                    cluster.add(elem)
                    return
            self.__clusters[key].append({elem})

    def get_clusters(self) -> FrozenSet[FrozenSet[T]]:
        result = set()
        for _, clusterset in self.__clusters.items():
            for cluster in clusterset:
                result.add(frozenset(cluster))
        return frozenset(result)