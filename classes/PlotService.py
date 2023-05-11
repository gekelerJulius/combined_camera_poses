from typing import Dict, List

from matplotlib.figure import Figure


class PlotService:
    plot_map: Dict[int, Figure] = {}

    @staticmethod
    def get_instance():
        if not hasattr(PlotService, "_instance"):
            PlotService._instance = PlotService()
        return PlotService._instance

    def __init__(self):
        if not hasattr(PlotService, "_instance"):
            PlotService._instance = self
        else:
            raise Exception("You cannot create another PlotService class")

    def add_plot(self, figure: Figure, plot_id: int = None) -> int:
        if plot_id is None:
            plot_id = self.get_next_plot_id()
        self.plot_map[plot_id] = figure
        return plot_id

    def get_plot(self, plot_id: int) -> Figure:
        return self.plot_map[plot_id]

    def remove_plot(self, plot_id: int):
        del self.plot_map[plot_id]

    def get_plot_ids(self) -> List[int]:
        return list(self.plot_map.keys())

    def get_plot_count(self) -> int:
        return len(self.plot_map)

    def clear_plots(self):
        self.plot_map.clear()

    def get_plot_map(self) -> Dict[int, Figure]:
        return self.plot_map

    def get_next_plot_id(self) -> int:
        return max(self.plot_map.keys()) + 1 if self.plot_map else 0
