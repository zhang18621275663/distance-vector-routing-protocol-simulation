# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'window.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1015, 601)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.Image = QtWidgets.QLabel(self.centralwidget)
        self.Image.setText("")
        self.Image.setObjectName("Image")
        self.horizontalLayout.addWidget(self.Image)
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.PointList = QtWidgets.QComboBox(self.centralwidget)
        self.PointList.setObjectName("PointList")
        self.verticalLayout.addWidget(self.PointList)
        self.Table = QtWidgets.QTableWidget(self.centralwidget)
        self.Table.setMinimumSize(QtCore.QSize(260, 0))
        self.Table.setObjectName("Table")
        self.Table.setColumnCount(0)
        self.Table.setRowCount(0)
        self.verticalLayout.addWidget(self.Table)
        self.verticalLayout.setStretch(0, 2)
        self.verticalLayout.setStretch(1, 8)
        self.horizontalLayout.addLayout(self.verticalLayout)
        self.horizontalLayout.setStretch(0, 9)
        self.horizontalLayout.setStretch(1, 2)
        self.gridLayout.addLayout(self.horizontalLayout, 0, 0, 1, 1)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setMinimumSize(QtCore.QSize(0, 40))
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout_2.addWidget(self.label_2)
        self.NodeNumber = QtWidgets.QSpinBox(self.centralwidget)
        self.NodeNumber.setMinimumSize(QtCore.QSize(0, 40))
        self.NodeNumber.setMinimum(3)
        self.NodeNumber.setMaximum(10)
        self.NodeNumber.setObjectName("NodeNumber")
        self.horizontalLayout_2.addWidget(self.NodeNumber)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem)
        self.ResetButton = QtWidgets.QPushButton(self.centralwidget)
        self.ResetButton.setMinimumSize(QtCore.QSize(0, 40))
        self.ResetButton.setObjectName("ResetButton")
        self.horizontalLayout_2.addWidget(self.ResetButton)
        self.NextButton = QtWidgets.QPushButton(self.centralwidget)
        self.NextButton.setMinimumSize(QtCore.QSize(0, 40))
        self.NextButton.setObjectName("NextButton")
        self.horizontalLayout_2.addWidget(self.NextButton)
        self.TerminalButton = QtWidgets.QPushButton(self.centralwidget)
        self.TerminalButton.setMinimumSize(QtCore.QSize(0, 40))
        self.TerminalButton.setObjectName("TerminalButton")
        self.horizontalLayout_2.addWidget(self.TerminalButton)
        self.horizontalLayout_2.setStretch(0, 2)
        self.horizontalLayout_2.setStretch(1, 2)
        self.horizontalLayout_2.setStretch(2, 1)
        self.horizontalLayout_2.setStretch(3, 2)
        self.horizontalLayout_2.setStretch(4, 2)
        self.horizontalLayout_2.setStretch(5, 2)
        self.gridLayout.addLayout(self.horizontalLayout_2, 1, 0, 1, 1)
        self.gridLayout.setRowMinimumHeight(0, 9)
        self.gridLayout.setRowMinimumHeight(1, 1)
        self.gridLayout.setRowStretch(0, 9)
        self.gridLayout.setRowStretch(1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label_2.setText(_translate("MainWindow", "NodeNumber："))
        self.ResetButton.setText(_translate("MainWindow", "Reset"))
        self.NextButton.setText(_translate("MainWindow", "Next"))
        self.TerminalButton.setText(_translate("MainWindow", "Terminal"))

