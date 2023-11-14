import tkinter
import tkinter as tk
from tkinter import messagebox, Canvas, Toplevel, PhotoImage, Label
import cv2
import threading
from image import ImageProcessor
import os
import keras
import numpy as np
from PIL import Image, ImageTk

import sys

model_path = "../model/model_sep.h5"
model_sep = keras.models.load_model(os.path.abspath(model_path))


def img_prediction(img):
    label = ['1 : 0', '2 : 1', '3 : 2', '4 : 3', '5 : 4',
             'down : 5', 'left : 6', 'right : 7', 'up : 8']
    prediction_val = model_sep.predict(np.expand_dims(img, axis=0))
    predicted_class = np.argmax(prediction_val)
    predicted_class_label = label[predicted_class].split(' : ')[0]

    # 인식된 결과와, 예측률을 반환
    return predicted_class_label, np.max(prediction_val)

def show_warning():
    messagebox.showwarning("경고", "이동할 수 없습니다.")



block_label = None
# x, y = 1, 1  # x와 y를 전역 변수로 초기화

def block_move():
    global block_label

    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        while True:
            ret, frame = cap.read()
            
            if ret:
                flipped_frame = cv2.flip(frame, -1)
                cv2.imshow('camera', flipped_frame)

                key = cv2.waitKey(1)
                
                if key != -1:
                    if key == 27:
                        break
                    elif key == 32:
                        cv2.imwrite('../move_img/photo.jpg', flipped_frame)
                        print("촬영을 완료했습니다.")

                        # 촬영된 이미지를 전처리
                        processed_img = ImageProcessor().img_processed(flipped_frame)
                        print('이미지 전처리 완료')

                        # 이미지에서 숫자 영역과 방향 영역을 따로 구분함
                        processed_dir = ImageProcessor().img_crop_processed(processed_img, 154, 170, 340, 250)
                        processed_num = ImageProcessor().img_crop_processed(processed_img, 500, 170, 175, 250)
                        print("이미지 분리 완료")

                        predictions_dir, pre_value_dir = img_prediction(processed_dir)
                        predictions_num, pre_value_num = img_prediction(processed_num)
                        print("이미지 예측 완료")
                        block_name = predictions_dir + ' ' + predictions_num
                        block_label = block_name

                        print("예측 클래스:", block_name)

                        print(pre_value_dir, pre_value_num)

                        # 예측률이 90% 미만일 경우 에러 메세지 반환
                        if pre_value_dir < 0.9 or pre_value_num < 0.9:
                            block_list = ['left 1', 'left 2', 'left 3', 'left 4', 'left 5',
                                          'right 1', 'right 2', 'right 3', 'right 4', 'right 5',
                                          'down 1', 'down 2', 'down 3', 'down 4', 'down 5',
                                          'up 1', 'up 2', 'up 3', 'up 4', 'up 5']
                            if block_name in block_list:
                                event_image = cv2.imread(f'../event_img/{block_name}.jpg')
                                cv2.imshow('camera', event_image)

                                while True:
                                    key_overlay = cv2.waitKey(1)
                                    if key_overlay == ord('y'):
                                        break
                                    elif key_overlay == ord('n'):
                                        break
                                
                                cv2.imshow('camera', flipped_frame)
                            # 예측률이 낮으며 인식 자체가 이상할 경우 다시 찍으라는 메세지 반환
                            else:
                                print("블럭 인식 안됨")
                                event_image = cv2.imread(f'../event_img/noblock.jpg')
                                cv2.imshow('camera', event_image)
                                key_overlay = cv2.waitKey(0)

                                if key_overlay == ord('y'):
                                    cv2.imshow('camera', flipped_frame)
            else:
                print('프레임이 없습니다')
                break
    else:
        print('카메라를 찾을 수 없습니다')

    cap.release()
    cv2.destroyAllWindows()

    return block_label


def create_window():
    def move_player_and_draw_line(dx, dy):
        global x, y, player, line_coords
        canvas.move(player, dx * 70, dy * 70)
        x += dx
        y += dy
        x1, y1 = x * 70 + 35, y * 70 + 35
        x2, y2 = (x - dx) * 70 + 35, (y - dy) * 70 + 35
        line_coords.append((x1, y1, x2, y2))
        canvas.create_line(x1, y1, x2, y2, fill="blue", width=3)
        

    def player_move(event):
        global x, y, block_label, maze
        print("player_move")

        dir = block_label.split(' ')[0]
        num = int(block_label.split(' ')[1])

        if dir == "up" :
            block = 0
            for i in range(1,int(num)+1):
                if maze[y-i][x] == 1:
                    block = 1
                    print("이동불가")
                    # 막혀있어 못간다는걸 이미지로 표현
                    show_warning()
            if block == 0:
                move_player_and_draw_line(0, -1*num)
                y -= 1*num

        if dir == "down":
            block = 0
            for i in range(1,int(num)+1):
                if maze[y+i][x] == 1:
                    block = 1
                    print("이동불가")
                    # 막혀있어 못간다는걸 이미지로 표현
                    show_warning()
            if block == 0:
                move_player_and_draw_line(0, 1*num)
                y += 1*num

        if dir == "left" :
            block = 0
            for i in range(1,int(num)+1):
                if maze[y][x-i] == 1:
                    block = 1
                    print("이동불가")
                    # 막혀있어 못간다는걸 이미지로 표현
                    show_warning()
            if block == 0:
                move_player_and_draw_line(-1*num, 0)
                x -= 1*num

        if dir == "right":
            block = 0
            for i in range(1,int(num)+1):
                print("for",i)
                if maze[y][x+i] == 1:
                    block = 1
                    print("이동불가")
                    # 막혀있어 못간다는걸 이미지로 표현
                    show_warning()
            if block == 0:
                move_player_and_draw_line(1*num, 0)
                x += 1*num

        if maze[y][x] == 2:
            messagebox.showinfo("미션 성공", "목적지에 도착! 축하합니다.")

    
    def on_button_click(new_maze, end_image_path, end_coords):
        global maze
        maze = new_maze
        redraw_maze(maze, end_image_path, end_coords)

    def on_button_click_tutorial():
        global photo
        image_path = "../event_img/tutorial.png"  # 이미지 경로를 적절하게 수정
        
        original_image = Image.open(image_path)
        resized_image = original_image.resize((700, 490), Image.ANTIALIAS)
        
        photo = ImageTk.PhotoImage(resized_image, master=window)

        # 이미지 생성
        canvas.delete("all")
        canvas.create_image(canvas.winfo_width() // 2, canvas.winfo_height() // 2, image=photo)

    

    def redraw_maze(maze, end_image_path, end_coords):
        global img_end, player_img, player, x, y, line_coords
        canvas.delete("all")
        for y in range(7):
            for x in range(10):
                if maze[y][x] == 1:
                    canvas.create_rectangle(x * 70, y * 70, (x + 1) * 70, (y + 1) * 70, fill="skyblue", outline="white")
        img_end = tk.PhotoImage(file=end_image_path, master=window)
        canvas.create_image(*end_coords, image=img_end)
        player_img = tk.PhotoImage(file="./리 공부해 (1).png", master=window)
        player = canvas.create_image(105, 105, image=player_img)
        x = 1
        y = 1
        line_coords = []

    def create_button(frame, text, command):
        button = tk.Button(frame, text=text, command=command)
        button.pack(side=tk.LEFT)
        return button
    

    def maze_make(maze_no):
        global maze
        if maze_no == 1:
            maze_re = [
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 0, 0, 0, 0, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 0, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 0, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 0, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 0, 0, 0, 0, 2, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                ]
        elif maze_no == 2:
            maze_re = [
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 0, 1, 1, 0, 0, 0, 0, 0, 1],
                [1, 0, 1, 1, 0, 1, 1, 1, 0, 1],
                [1, 0, 1, 1, 0, 1, 1, 1, 0, 1],
                [1, 0, 1, 1, 0, 1, 1, 1, 0, 1],
                [1, 0, 0, 0, 0, 1, 1, 1, 2, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                ]
        else:
            maze_re = [
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 0, 0, 1, 1, 1, 2, 0, 0, 1],
                [1, 1, 0, 1, 1, 1, 1, 1, 0, 1],
                [1, 0, 0, 1, 0, 0, 0, 1, 0, 1],
                [1, 0, 1, 0, 0, 1, 0, 1, 0, 1],
                [1, 0, 0, 0, 1, 1, 0, 0, 0, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                ]
        
        maze = maze_re
        return maze



    maze = maze_make(1)
    line_coords = []
    window = tkinter.Tk()
    window.title("k-OSMO")
    frame_canvas = tk.Frame(window)
    frame_canvas.pack()
    canvas = tk.Canvas(frame_canvas, width=700, height=490, bg="white")
    canvas.pack()
    


    
   
    
    


    
    
    end_image_path = "./리 공부해 (1).png"
    end_coords1 = (590, 385)
    end_coords2 = (590, 385)
    end_coords3 = (455, 105)
    redraw_maze(maze, end_image_path, end_coords1)

    
    canvas.bind_all("<KeyPress>", player_move)
    frame_buttons = tk.Frame(window)
    frame_buttons.pack()
    create_button(frame_buttons, "1단계", lambda: on_button_click(maze_make(1), end_image_path, end_coords1))
    create_button(frame_buttons, "2단계", lambda: on_button_click(maze_make(2), end_image_path, end_coords2))
    create_button(frame_buttons, "3단계", lambda: on_button_click(maze_make(3), end_image_path, end_coords3))
    create_button(frame_buttons, "듀토리얼", on_button_click_tutorial)
    window.mainloop()


if __name__ == "__main__":
    # Start the camera thread
    camera_thread = threading.Thread(target=block_move)
    camera_thread.start()

    # Start the maze window in the main thread
    create_window()