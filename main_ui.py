import sys
from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QPushButton,
    QVBoxLayout, QHBoxLayout, QFileDialog, QDialog
)

from PySide6 import QtCore, QtGui

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("乐乐课堂视频爬虫")
        
        # A label says: '链接:'
        link_label = QLabel("链接:             ")
        # A line edit field to input url
        #   With a gray hint telling user to put in url
        self.link_edit = QLineEdit()
        self.link_edit.setPlaceholderText('请输入要爬取的网页链接')
        # Assign these two to a QHBoxLayout
        link_layout = QHBoxLayout()
        link_layout.addWidget(link_label)
        link_layout.addWidget(self.link_edit)

        # A label says: '输出文件夹目录:'
        output_label = QLabel("文件输出目录: ")
        # A line edit to input path to store the files
        self.output_edit = QLineEdit()
        self.output_edit.setPlaceholderText('若留空则默认存放为当前目录下')
        # A button to open up file explorer
        output_btn = QPushButton("...")
        output_btn.clicked.connect(self.chooseFolder)
        # Assign these widgets to a QHBoxLayout
        output_layout = QHBoxLayout()
        output_layout.addWidget(output_label)
        output_layout.addWidget(self.output_edit)
        output_layout.addWidget(output_btn)

        # A label says: '下载间隔区间:'
        range_label = QLabel("下载间隔区间: ")
        # Two saparated line edit splitting by a label: '-'
        #   Only number allowed
        #   if the left text is larger than the right one, automatically adjust the right one
        self.range_l = QLineEdit()
        self.range_l.setPlaceholderText('仅数字')
        self.range_l.setValidator(QtGui.QDoubleValidator())
        self.range_l.textChanged.connect(self.checkTimeInterval)
        split_label = QLabel(" - ")
        self.range_r = QLineEdit()
        self.range_r.setPlaceholderText('仅数字')
        self.range_r.setValidator(QtGui.QDoubleValidator())
        self.range_r.textChanged.connect(self.checkTimeInterval)
        sub_layout = QHBoxLayout()
        sub_layout.addWidget(self.range_l)
        sub_layout.addWidget(split_label)
        sub_layout.addWidget(self.range_r)
        # Assign these widgets to a QHBoxLayout
        range_layout = QHBoxLayout()
        range_layout.addWidget(range_label)
        range_layout.addStretch(1)
        range_layout.addLayout(sub_layout)
        range_layout.addStretch(1)

        # A button to submit all information
        submit_btn = QPushButton("提交")
        submit_btn.clicked.connect(self.submit)

        # Assign all above to a QVBoxLayout
        window_layout = QVBoxLayout()
        window_layout.addLayout(link_layout)
        window_layout.addLayout(output_layout)
        window_layout.addLayout(range_layout)
        window_layout.addWidget(submit_btn)
        

        # Apply this layout
        self.setLayout(window_layout)

    @QtCore.Slot()
    def chooseFolder(self):
        d_path = QFileDialog().getExistingDirectory()
        self.output_edit.setText(d_path)

    @QtCore.Slot()
    def checkTimeInterval(self):
        l = self.range_l.text()
        r = self.range_r.text()
        
        if l and r:
            if float(l) > float(r):
                if self.range_r.hasFocus(): self.range_l.setText(str(float(r) - 1) if float(r) >= 1 else "0")
                elif self.range_l.hasFocus(): self.range_r.setText(str(float(l) + 1))

    @QtCore.Slot()
    def submit(self):
        t_range = (float(self.range_l.text()), float(self.range_r.text()))

        print(self.link_edit.text())
        print(self.output_edit.text())
        print(t_range)

class PopupWindow(QDialog):
    def __init__(self):
        super().__init__()
        self.resize(300, 240)

def test():
    app = QApplication([])
    window = MainWindow()
    window.resize(400, 300)
    window.setFocus()
    window.show()

    app.exec()

if __name__ == '__main__':
    test()