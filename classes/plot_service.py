from typing import Dict, List

from matplotlib.figure import Figure


class PlotService:
    plot_map: Dict[str, Figure] = {}

    @staticmethod
    def get_instance():
        if not hasattr(PlotService, "_instance"):
            PlotService._instance = PlotService()
        return PlotService._instance

    def __init__(self):
        if not hasattr(PlotService, "_instance"):
            PlotService._instance = self
        else:
            raise Exception("You cannot create another PlotService class, use PlotService.get_instance() instead")

    def add_plot(self, figure: Figure, plot_id: str) -> str:
        self.plot_map[plot_id] = figure
        return plot_id

    def get_plot(self, plot_id: str) -> Figure:
        return self.plot_map.get(plot_id)

    def remove_plot(self, plot_id: str):
        del self.plot_map[plot_id]

    def get_plot_ids(self) -> List[str]:
        return list(self.plot_map.keys())

    def get_plot_count(self) -> int:
        return len(self.plot_map)

    def clear_plots(self):
        self.plot_map.clear()

    def get_plot_map(self) -> Dict[str, Figure]:
        return self.plot_map

    def plot_exists(self, plot_id):
        return plot_id in self.plot_map
