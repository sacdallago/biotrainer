from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TypeAlias

from pydantic import BaseModel, Field

# Type aliases for clarity, matching Dart's Score and Place
Score: TypeAlias = float
Place: TypeAlias = int


class RankingEntry(BaseModel):
    name: str
    metrics: Dict[str, Any]

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, RankingEntry):
            return False
        return self.name == other.name


class RankingGroup(BaseModel):
    name: str
    group_function: Callable[[Set[str]], Set[str]]


class _RankingResult(BaseModel):
    category_ranking_map: Dict[str, Dict[RankingEntry, Score]]
    group_ranking_map: Optional[Dict[str, Dict[RankingEntry, Score]]] = None
    leaderboard: Dict[RankingEntry, Score] = Field(default_factory=dict)
    calculated_leaderboard_ranking: List[Tuple[Place, RankingEntry, Score]] = Field(
        default_factory=list
    )

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **data: Any):
        super().__init__(**data)
        if self.leaderboard and not self.calculated_leaderboard_ranking:
            # Use object.__setattr__ to bypass Pydantic's immutability if needed,
            # but here we are in __init__
            self.calculated_leaderboard_ranking = self.calculate_leaderboard_ranking(
                self.leaderboard
            )

    @staticmethod
    def calculate_leaderboard_ranking(
        leaderboard: Dict[RankingEntry, Score]
    ) -> List[Tuple[Place, RankingEntry, Score]]:
        # Sort by score descending (highest first)
        sorted_entries = sorted(
            leaderboard.items(), key=lambda item: item[1], reverse=True
        )

        result: List[Tuple[Place, RankingEntry, Score]] = []
        current_place: Place = 1
        previous_score: Optional[Score] = None

        for i, (entry, current_score) in enumerate(sorted_entries):
            # If score is different from previous, update place to current index + 1
            if previous_score is not None and current_score != previous_score:
                current_place = i + 1

            result.append((current_place, entry, current_score))
            previous_score = current_score

        return result

    def get_place(self, entry_name: str) -> int:
        for place, entry, _ in self.calculated_leaderboard_ranking:
            if entry.name == entry_name:
                return place
        raise ValueError(f"Entry {entry_name} not found in ranking")

    def get_score(self, entry_name: str) -> float:
        for entry, score in self.leaderboard.items():
            if entry.name == entry_name:
                return score
        return 0.0

    @property
    def ranking_map(self) -> Dict[str, Dict[RankingEntry, Score]]:
        """Get the actual ranking map that is used for the leaderboard calculation"""
        return (
            self.group_ranking_map
            if self.group_ranking_map is not None
            else self.category_ranking_map
        )

    @property
    def number_competitors(self) -> int:
        return len(self.leaderboard)

    @property
    def ranking_categories(self) -> Set[str]:
        return set(self.ranking_map.keys())

    @property
    def number_categories(self) -> int:
        return len(self.ranking_categories)


class Ranking:
    def __init__(self, result: _RankingResult, applied_weights: Dict[str, int]):
        self._result = result
        self._applied_weights = applied_weights

    @classmethod
    def calculate(
        cls,
        entries: List[RankingEntry],
        groups: Optional[List[RankingGroup]] = None,
        weights: Optional[Dict[str, int]] = None,
        is_ascending_metric: Optional[Callable[[str], bool]] = None,
    ) -> Ranking:
        # Set ascending to False by default
        if is_ascending_metric is None:

            def is_ascending_metric(_: str) -> bool:
                return False

        category_ranking_result = cls._calculate_category_ranking(
            entries, is_ascending_metric
        )

        # Apply groups to calculated rankings
        group_ranking_result = cls._calculate_group_ranking(
            groups, category_ranking_result
        )

        # Set weights to 0 by default
        if weights is None:
            weights = {cat: 0 for cat in group_ranking_result.ranking_map.keys()}

        # Calculate leaderboard from rankings with weights
        leaderboard_ranking_result = cls._calculate_leaderboard_with_weights(
            group_ranking_result, weights
        )

        return cls(leaderboard_ranking_result, weights)

    @staticmethod
    def _calculate_category_ranking(
        entries: List[RankingEntry], is_ascending_metric: Callable[[str], bool]
    ) -> _RankingResult:
        # Get categories
        categories: Set[str] = set()
        for entry in entries:
            categories.update(entry.metrics.keys())

        # Validate that all entries have the same categories
        for category in categories:
            for entry in entries:
                if category not in entry.metrics:
                    raise ValueError(
                        f"Found a category ({category}) that is not existent in all ranking entries!"
                    )

        # Calculate rankings for given categories
        category_ranking: Dict[str, Dict[RankingEntry, Score]] = {}
        for category in categories:
            category_ranking[category] = Ranking._calculate_single_category_ranking(
                category, entries, is_ascending_metric(category)
            )
        return _RankingResult(category_ranking_map=category_ranking)

    @staticmethod
    def _calculate_single_category_ranking(
        category: str, entries: List[RankingEntry], is_ascending: bool
    ) -> Dict[RankingEntry, Score]:
        # Sort entries by metric value
        # Python's sort is stable
        sorted_entries = sorted(entries, key=lambda e: e.metrics[category])

        if is_ascending:
            sorted_entries.reverse()

        result: Dict[RankingEntry, Score] = {}
        if not sorted_entries:
            return result

        current_rank = 1
        same_rank_count = 0

        # Handle first entry
        previous_score = sorted_entries[0].metrics[category]
        result[sorted_entries[0]] = float(current_rank)

        # Process remaining entries
        for i in range(1, len(sorted_entries)):
            current_score = sorted_entries[i].metrics[category]

            if current_score == previous_score:
                # Same score as previous entry, assign same rank
                result[sorted_entries[i]] = float(current_rank)
                same_rank_count += 1
            else:
                # Different score, assign next rank (skip ranks for ties)
                current_rank += same_rank_count + 1
                result[sorted_entries[i]] = float(current_rank)
                same_rank_count = 0

            previous_score = current_score

        return result

    @staticmethod
    def _calculate_group_ranking(
        groups: Optional[List[RankingGroup]], category_ranking: _RankingResult
    ) -> _RankingResult:
        if groups is None:
            return category_ranking

        categories = category_ranking.ranking_categories
        # Copy the nested map
        group_ranking_map = {
            k: v.copy() for k, v in category_ranking.category_ranking_map.items()
        }

        for group in groups:
            categories_to_group = group.group_function(categories)
            # Check that all categories exist
            for category_to_group in categories_to_group:
                if category_to_group not in group_ranking_map:
                    raise ValueError(
                        f"Did not find group {category_to_group} in existing categories!"
                    )

            rankings_to_average = [
                group_ranking_map[cat] for cat in categories_to_group
            ]
            averaged_ranking = Ranking._get_rankings_average(rankings_to_average)

            group_ranking_map[group.name] = averaged_ranking
            for cat in categories_to_group:
                del group_ranking_map[cat]

        return _RankingResult(
            category_ranking_map=category_ranking.category_ranking_map,
            group_ranking_map=group_ranking_map,
        )

    @staticmethod
    def _get_rankings_average(
        rankings: List[Dict[RankingEntry, Score]]
    ) -> Dict[RankingEntry, float]:
        if not rankings:
            return {}

        result: Dict[RankingEntry, float] = {}
        number_rankings = len(rankings)

        for ranking in rankings:
            for entry, score in ranking.items():
                result[entry] = result.get(entry, 0.0) + score

        return {entry: score / number_rankings for entry, score in result.items()}

    @staticmethod
    def _calculate_leaderboard_with_weights(
        group_ranking: _RankingResult, weights: Dict[str, int]
    ) -> _RankingResult:
        ranking_map = group_ranking.ranking_map
        leaderboard: Dict[RankingEntry, float] = {}

        for category_or_group_name, ranking in ranking_map.items():
            multiplier = Ranking.get_score_multiplier(
                weights.get(category_or_group_name, 0)
            )
            for entry, score in ranking.items():
                weighted_score = score * multiplier
                leaderboard[entry] = leaderboard.get(entry, 0.0) + weighted_score

        return _RankingResult(
            category_ranking_map=group_ranking.category_ranking_map,
            group_ranking_map=group_ranking.group_ranking_map,
            leaderboard=leaderboard,
        )

    @staticmethod
    def get_score_multiplier(weight: int) -> float:
        return 1.0 + (weight / 10.0)

    def update_weights(self, updated_weights: Dict[str, int]) -> Ranking:
        new_leaderboard_result = self._calculate_leaderboard_with_weights(
            self._result, updated_weights
        )
        return Ranking(new_leaderboard_result, updated_weights)

    @staticmethod
    def _format_ranking_score(value: float) -> str:
        # In Python, float('inf') is often used instead of double.maxFinite
        if value == float("inf"):
            return "Infinite"
        return f"{value:.1f}"

    def verbose_ranking_by_entry(self, entry_name: str) -> Optional[str]:
        ranking_entry: Optional[RankingEntry] = None
        for entry in self._result.leaderboard.keys():
            if entry.name == entry_name:
                ranking_entry = entry
                break

        if ranking_entry is None:
            return None

        leaderboard_place = self._result.get_place(entry_name)
        total_score = self._format_ranking_score(self._result.get_score(entry_name))
        n_competitors = int(self._result.number_competitors)
        max_category_score = self._format_ranking_score(self.max_category_score)
        maximum_ranking_value = self._format_ranking_score(self.maximum_ranking_value)
        verbose_lines = []
        for category, ranking in self._result.ranking_map.items():
            # Find entry score in this category
            entry_score = 0.0
            for e, s in ranking.items():
                if e.name == entry_name:
                    entry_score = s
                    break

            metric_val = ranking_entry.metrics.get(category)
            metric_str = f"Metric: {metric_val}" if metric_val is not None else "Metric: combined task mean"
            verbose_lines.append(f"{category}: \n\tScore: {entry_score}/{max_category_score} \n\t{metric_str}")

        verbose_rank_string = "\n".join(verbose_lines)

        return (
            f"{entry_name}:\n"
            f"Global Position: {leaderboard_place}. Place\n\n"
            f"Categories: \n"
            f"{verbose_rank_string}\n\n"
            f"Number of competitors: {n_competitors}\n"
            f"Number of categories: {self._result.number_categories}\n"
            f"Total score: {total_score}/{maximum_ranking_value}"
        )

    def copied_ranking(self) -> str:
        results = []
        for entry in self._result.leaderboard.keys():
            verbose = self.verbose_ranking_by_entry(entry.name)
            if verbose:
                results.append(verbose)
        return "\n\n".join(results)

    def get_place(self, entry_name: str) -> int:
        return self._result.get_place(entry_name)

    def get_score(self, entry_name: str) -> float:
        return self._result.get_score(entry_name)

    @property
    def max_category_score(self) -> float:
        return float(self._result.number_competitors)

    @property
    def maximum_ranking_value(self) -> float:
        max_category_score = self.max_category_score
        return sum(
            max_category_score * self.get_score_multiplier(self._applied_weights.get(cat, 0))
            for cat in self._result.ranking_categories
        )

    @property
    def categories(self) -> List[str]:
        return list(self._result.ranking_map.keys())

    def get_leaderboard_ranking(
        self, ascending: bool = False
    ) -> List[Tuple[Place, RankingEntry, Score]]:
        ranking = self._result.calculated_leaderboard_ranking
        return ranking[::-1] if ascending else ranking

    def get_category_ranking(
        self, category: str, ascending: bool = False
    ) -> Optional[List[Tuple[Place, RankingEntry, Score]]]:
        category_map = self._result.category_ranking_map.get(category) or self._result.group_ranking_map.get(category)
        if category_map is None:
            return None
        category_ranking = _RankingResult.calculate_leaderboard_ranking(category_map)
        return category_ranking[::-1] if ascending else category_ranking

    @property
    def ranking_categories(self) -> Set[str]:
        return self._result.ranking_categories

    @property
    def raw_categories(self) -> Set[str]:
        """Categories before grouping was applied"""
        return set(self._result.category_ranking_map.keys())
