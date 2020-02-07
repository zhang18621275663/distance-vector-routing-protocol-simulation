import sys
import cv2
import random
import hashlib
import networkx as nx
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from window import Ui_MainWindow
from itertools import combinations, permutations


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parents=None):
        super(MainWindow, self).__init__(parents)
        self.setupUi(self)
        self.setStyleSheet("#MainWindow{background-color: white}")
        self.showFullScreen()
        self.router_table = dict()
        self.router_map = None

        self.reset_matrix()

        self.Table.setColumnCount(3)
        self.Table.setHorizontalHeaderLabels(["Node", "Distance", "NextHop"])
        self.Table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.ResetButton.clicked.connect(self.reset_matrix)

        # bind combobox change event
        self.PointList.currentIndexChanged.connect(self.node_list_change_event)

        # bind single update
        self.NextButton.clicked.connect(self.single_update)

        # bind multi update
        self.TerminalButton.clicked.connect(self.multi_update)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_A:
            self.showFullScreen()
        if event.key() == Qt.Key_Escape:
            self.showNormal()

    def node_list_change_event(self):
        self.Table.clearContents()
        current_index_str = self.PointList.currentText()
        if current_index_str.strip():
            current_index = int(current_index_str.replace("Node", "").strip())
            node_router_table = self.router_table.get(current_index)
            if node_router_table:
                self.Table.setRowCount(len(node_router_table))
                for index, (node, distance, next_hop) in enumerate(node_router_table):
                    new_view_node = QTableWidgetItem(str(node))
                    new_view_node.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
                    new_view_distance = QTableWidgetItem(str(distance))
                    new_view_distance.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
                    new_view_next_htop = QTableWidgetItem(str(next_hop))
                    new_view_next_htop.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)

                    self.Table.setItem(index, 0, new_view_node)
                    self.Table.setItem(index, 1, new_view_distance)
                    self.Table.setItem(index, 2, new_view_next_htop)

    def reset_matrix(self):
        self.PointList.clear()
        self.router_table = dict()
        node_number = self.NodeNumber.text()
        node_number = int(node_number)
        self.reset_combox(node_number)
        self.router_map, image = self.generate_router_matrix(node_number, (800, 800))
        showImage = QImage(image, image.shape[1], image.shape[0], QImage.Format_RGB888)
        self.Image.setPixmap(QPixmap.fromImage(showImage))
        self.Image.setAlignment(Qt.AlignCenter)

    def generate_router_matrix(self, points_number, image_size):
        def compute_distance(a, b):
            return ((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2) ** 0.5

        points_list = list()
        inner_width, inner_height = int(image_size[1] / 5 * 4), int(image_size[0] / 5 * 4)

        count = 0
        point_saved_thres = 2
        while count < points_number:
            x, y = random.randint(0, inner_height), random.randint(0, inner_width)
            if len(points_list) == 0:
                points_list.append((x, y))
                count += 1
            else:
                use_flag = True
                for point in points_list:
                    dis = int(compute_distance((x, y), point) / 50)
                    if dis < point_saved_thres:
                        use_flag = False
                if use_flag:
                    points_list.append((x, y))
                    count += 1

        router_map = np.zeros((points_number, points_number))
        for i in range(points_number):
            for j in range(points_number):
                router_map[i, j] = compute_distance(points_list[i], points_list[j])

        mask = np.zeros((points_number, points_number))

        # 计算距离求mask
        dis_list = router_map.copy().flatten()
        dis_list[dis_list == 0] = 800 * 800
        index_list = dis_list.argsort()

        lines_number = points_number - 1 + random.randint(0, int(points_number / 3))
        index_list = index_list[:lines_number * 2]
        for lines_index in index_list:
            i = int(lines_index / points_number)
            j = lines_index % points_number
            mask[i, j] = 1
            mask[j, i] = 1

        # 连接不连通子图
        mask = self.find_sub_graph(mask, router_map)
        router_map = router_map / 50
        router_map = router_map * mask
        router_map = router_map.astype(np.int)

        # point lines
        font = cv2.FONT_HERSHEY_SIMPLEX

        img = np.ones((image_size[0], image_size[1], 3), np.uint8) * 255
        for i in range(points_number):
            for j in range(points_number):
                if j > i and router_map[i, j] > 0:
                    point_1 = (points_list[i][0] + int((image_size[0] - inner_height) / 2),
                               points_list[i][1] + int((image_size[1] - inner_width) / 2))

                    point_2 = (points_list[j][0] + int((image_size[0] - inner_height) / 2),
                               points_list[j][1] + int((image_size[1] - inner_width) / 2))

                    cv2.line(img, point_1, point_2, (255, 150, 10), 2)
                    text_x = int(abs((point_2[0] + point_1[0]) / 2))
                    text_y = int(abs((point_2[1] + point_1[1]) / 2))
                    cv2.putText(img, str(router_map[i, j]), (text_x, text_y),
                                font, 0.5, (0, 0, 255), 1)

        # plot points
        point_color = (0, 0, 0)  # BGR
        r = 20
        for index, point in enumerate(points_list):
            c_x = point[0] + int((image_size[0] - inner_height) / 2)
            c_y = point[1] + int((image_size[1] - inner_width) / 2)
            cv2.circle(img, (c_x, c_y), r, point_color, -1)
            cv2.circle(img, (c_x, c_y), r - 2, (255, 255, 255), -1)

            cv2.putText(img, str(index), (int(c_x - r / 2), int(c_y + r / 2)), font, 1, (0, 0, 0), 1)

        return router_map, img

    def reset_combox(self, point_numbers):
        for i in range(point_numbers):
            self.PointList.addItem(str("Node {}".format(i)))

    def single_update(self):
        if len(self.router_table.keys()) == 0:
            for i in range(self.router_map.shape[0]):
                for j in range(self.router_map.shape[1]):
                    if self.router_map[i, j] != 0:
                        item_list = self.router_table.get(i)
                        if not item_list:
                            item_list = [[j, self.router_map[i, j], j]]

                        else:
                            item_list.append([j, self.router_map[i, j], j])
                        self.router_table[i] = item_list
        else:
            for i in range(self.router_map.shape[0]):
                for j in range(self.router_map.shape[1]):
                    if self.router_map[i, j] != 0:
                        j_router_table = self.router_table[j]
                        i_j_weight = self.router_map[i, j]
                        for j_router in j_router_table:
                            j_node, j_distance, j_next_hop = j_router
                            if j_node != i:
                                # 查找本地路由表
                                i_router_table = self.router_table[i]

                                i_node_list = []
                                for i_router in i_router_table:
                                    i_node, i_distance, i_next_hop = i_router
                                    i_node_list.append(i_node)
                                if j_node in i_node_list:
                                    index = i_node_list.index(j_node)
                                    i_node, i_distance, i_next_hop = i_router_table[index]
                                    # 判断距离
                                    if j_distance + i_j_weight <= i_distance:
                                        self.router_table[i][index] = (j_node, j_distance + i_j_weight, j)
                                else:
                                    # 直接添加
                                    self.router_table[i].append((j_node, j_distance + i_j_weight, j))

        self.node_list_change_event()

    def multi_update(self):
        last_router_table = hashlib.md5(str(self.router_table).encode("utf-8")).hexdigest()
        while True:
            self.single_update()
            current_router_table = hashlib.md5(str(self.router_table).encode("utf-8")).hexdigest()
            if last_router_table != current_router_table:
                last_router_table = current_router_table
            else:
                break

    def find_sub_graph(self, mask, router_map):
        G = nx.Graph()
        for node in range(mask.shape[0]):
            G.add_node(node)

        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if mask[i, j] == 1:
                    G.add_edge(i, j)

        sub_set = list()
        for c in nx.connected_components(G):
            # 得到不连通的子集
            nodeSet = G.subgraph(c).nodes()
            sub_set.append(nodeSet)
        sub_set_number = len(sub_set)
        comb_list = list(combinations(list(range(sub_set_number)), 2))
        for index_1, index_2 in comb_list:
            set_1 = sub_set[index_1]
            set_2 = sub_set[index_2]
            min = 800 * 800
            min_v = None
            for ii in set_1:
                for jj in set_2:
                    if router_map[ii, jj] < min:
                        min = router_map[ii, jj]
                        min_v = (ii, jj)
            if min_v:
                mask[min_v[0], min_v[1]] = 1
                mask[min_v[1], min_v[0]] = 1
        return mask


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
