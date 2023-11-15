import tkinter
import tkinter as tk
from tkinter import messagebox
import cv2
import threading
from image import ImageProcessor
import os
import keras
import numpy as np
from PIL import Image, ImageTk
import time


model_path = "../model/model_sep.h5"
model_sep = keras.models.load_model(os.path.abspath(model_path))

def create_window():

    def img_prediction(img):
        label = ['1 : 0', '2 : 1', '3 : 2', '4 : 3', '5 : 4',
                'down : 5', 'left : 6', 'right : 7', 'up : 8']
        prediction_val = model_sep.predict(np.expand_dims(img, axis=0))
        predicted_class = np.argmax(prediction_val)
        predicted_class_label = label[predicted_class].split(' : ')[0]
        # 인식된 결과와, 예측률을 반환
        return predicted_class_label, np.max(prediction_val)

    def show_warning():
        # 이동할 수 없는 경로를 실행하고자 할 경우 경고 메세지를 반환
        messagebox.showwarning("경고", "이동할 수 없습니다.")

    block_label = None
    # x, y = 1, 1  # x와 y를 전역 변수로 초기화

    def block_move():
        global block_label
        cap = cv2.VideoCapture(0)
        
        if cap.isOpened():
            print("************************************")
            print("카메라에 180도 회전이 적용되어 있습니다.")
            print("************************************")
            while True:
                ret, frame = cap.read()
                
                if ret:
                    flipped_frame = cv2.flip(frame, -1)
                    flipped_frame = cv2.resize(flipped_frame,(900,675))
                    
                    cv2.imshow('camera', flipped_frame)

                    key = cv2.waitKey(1)
                    
                    if key != -1:
                        if key == 27: # esc
                            break
                        elif key == 32: # spacebar
                            
                            cv2.imwrite('../img_data/pending_img/original_img.jpg', flipped_frame)
                            print("촬영을 완료했습니다.")

                            # 촬영된 이미지를 전처리
                            processed_img = ImageProcessor().img_processed(flipped_frame)
                            print('이미지 전처리 완료')

                            # 이미지에서 숫자 영역과 방향 영역을 따로 구분함
                            processed_dir = ImageProcessor().img_crop_processed(processed_img, 154, 170, 340, 250)
                            processed_num = ImageProcessor().img_crop_processed(processed_img, 500, 170, 175, 250)
                            cv2.imwrite('../img_data/pending_img/direction_img.jpg', processed_dir) ## PI6
                            cv2.imwrite('../img_data/pending_img/number_img.jpg', processed_num) ## PI7
                            print("이미지 분리 완료")

                            predictions_dir, pre_value_dir = img_prediction(processed_dir) 
                            predictions_num, pre_value_num = img_prediction(processed_num)
                            
                            print("이미지 예측 완료")
                            block_name = predictions_dir + ' ' + predictions_num
                            block_label = block_name

                            print("예측 값:", block_name)

                            print(f"방향 예측확률 : {pre_value_dir}, 숫자 예측 확률 : {pre_value_num}")

                            # 예측률이 90% 미만일 경우 에러 메세지 반환
                            if pre_value_dir < 0.9 or pre_value_num < 0.9:
                                block_list = ['left 1', 'left 2', 'left 3', 'left 4', 'left 5',
                                            'right 1', 'right 2', 'right 3', 'right 4', 'right 5',
                                            'down 1', 'down 2', 'down 3', 'down 4', 'down 5',
                                            'up 1', 'up 2', 'up 3', 'up 4', 'up 5']
                                if block_name in block_list:
                                    event_image = cv2.imread(f'../event_img/{block_name}.jpg')
                                    event_image = cv2.resize(event_image,(900,675)) # 이미지 사이즈 변경
                                    cv2.imshow('camera', event_image)

                                    while True:
                                        key_overlay = cv2.waitKey(1)
                                        if key_overlay == ord('y'):
                                            
                                            player_move()
                                            break
                                        elif key_overlay == ord('n'):
                                            break
                                    
                                    cv2.imshow('camera', flipped_frame)
                                # 예측률이 낮으며 인식 자체가 이상할 경우 다시 찍으라는 메세지 반환
                                else:
                                    print("블럭 인식 안됨")
                                    event_image = cv2.imread(f'../event_img/no block.jpg')
                                    event_image = cv2.resize(event_image,(900,675)) # 이미지 사이즈 변경
                                    cv2.imshow('camera', event_image)
                                    key_overlay = cv2.waitKey(0)

                            else:
                                
                                player_move()
                        
                                
                else:
                    print('프레임이 없습니다')
                    break
        else:
            print('카메라를 찾을 수 없습니다')
       
        cap.release()
        cv2.destroyAllWindows()
        

        return block_label


    def player_move():
        global mx, my, maze, block_label, square_weight
        print("캐릭터가 이동합니다.")

        dir = block_label.split(' ')[0]
        num = int(block_label.split(' ')[1])
        # 방향값과 숫자가 들어오면 for 문을 순환하며 이동할 수 있는 곳인지 확인
        if dir == "up" :
            block = 0
            for i in range(1,int(num)+1):
                if maze[my-i][mx] == 1:
                    block = 1
                    print("이동불가")
                    # 이동이 불가할 경우 messagebox를 나타냄
                    show_warning()
            if block == 0:
                for i in range(0,int(num)):
                    canvas.create_rectangle(mx * square_weight +1, (my-i) * square_weight+1, mx * square_weight + square_weight-1, (my-i) * square_weight + square_weight-1, fill="pink", width=0, tag="PAINT")
                    canvas.move(player, 0, -1 * square_weight)
                    time.sleep(0.5)
                my -= 1*num

        if dir == "down":
            block = 0
            for i in range(1,int(num)+1):
                if maze[my+i][mx] == 1:
                    block = 1
                    print("이동불가")
                    show_warning()
            if block == 0:
                
                for i in range(0,int(num)):
                    canvas.create_rectangle(mx * square_weight +1, (my+i) * square_weight+1, mx * square_weight + square_weight-1, (my+i) * square_weight + square_weight-1, fill="pink", width=0, tag="PAINT")
                    canvas.move(player, 0, 1 * square_weight)
                    time.sleep(0.5)
                my += 1*num

        if dir == "left" :
            block = 0
            for i in range(1,int(num)+1):
                if maze[my][mx-i] == 1:
                    block = 1
                    print("이동불가")
                    show_warning()
            if block == 0:
                
                for i in range(0,int(num)):
                    canvas.create_rectangle((mx-i) * square_weight + 1, my * square_weight +1, (mx-i) * square_weight + square_weight -1, my * square_weight + square_weight-1, fill="pink", width=0, tag="PAINT")
                    canvas.move(player, -1 * square_weight, 0)
                    time.sleep(0.5)

                mx -= 1*num

        if dir == "right":
            block = 0
            for i in range(1,int(num)+1):
                if maze[my][mx+i] == 1:
                    block = 1
                    print("이동불가")
                    show_warning()
            if block == 0:

                for i in range(0,int(num)):
                    canvas.create_rectangle((mx+i) * square_weight +1, my * square_weight + 1, (mx+i) * square_weight + square_weight -1, my * square_weight + square_weight -1 , fill="pink", width=0, tag="PAINT")
                    canvas.move(player, 1 * square_weight, 0)
                    time.sleep(0.5)

                mx += 1*num


        if maze[my][mx] == 2:
            messagebox.showinfo("미션 성공", "목적지에 도착! 축하합니다.")

    def on_camera_button_click():
        camera_thread = threading.Thread(target=block_move)
        camera_thread.start()

    def on_button_click(new_maze, end_image_path, end_coords):
        global maze
        maze = new_maze
        redraw_maze(maze, end_image_path, end_coords)

    def on_button_click_tutorial():
        global photo,square_weight
        original_image = Image.open("../event_img/tutorial.png")
        # 이미지를 캔버스 사이즈에 맞게 변경
        resized_image = original_image.resize((square_weight * 10, square_weight* 7), Image.ANTIALIAS)
        photo = ImageTk.PhotoImage(resized_image, master=window)
        canvas.delete("all")
        # 이미지를 캔버스 정가운데 배치
        canvas.create_image((square_weight * 10)//2,(square_weight* 7)//2, image=photo)


    def redraw_maze(maze, end_image_path, end_coords):
        global img_end, player_img, player, mx, my, line_coords,square_weight
        canvas.delete("all")
        for y in range(7):
            for x in range(10):
                if maze[y][x] == 1:
                    canvas.create_rectangle(x * square_weight, y * square_weight, (x + 1) * square_weight, (y + 1) * square_weight, fill="skyblue", outline="white")
        img_end = tk.PhotoImage(file=end_image_path, master=window)
        canvas.create_image(*end_coords, image=img_end)
        player_img = tk.PhotoImage(file="../event_img/ai_img.png", master=window)
        player = canvas.create_image(square_weight*1.5, square_weight*1.5, image=player_img)
        mx = 1
        my = 1
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
    global square_weight
    square_weight = 90 # 한칸당 할당된 너비
    maze = maze_make(1)
    line_coords = []
    window = tkinter.Tk()
    window.title("k-OSMO")
    
    frame_canvas = tk.Frame(window)
    frame_canvas.pack()
    canvas = tk.Canvas(frame_canvas, width=square_weight*10, height=square_weight*7, bg="white")
    canvas.pack()
    # 열리는 위치 지정
    window.geometry(f"{square_weight*10}x{square_weight*7+30}+10+10")
    

    end_image_path = "../event_img/end_img.png"
    # 미로별 끝나는 지점의 좌표
    end_coords1 = (8*square_weight + square_weight//2 , 5*square_weight+square_weight//2)
    end_coords2 = (8*square_weight + square_weight//2, 5*square_weight + square_weight//2)
    end_coords3 = (6*square_weight + square_weight//2, 1*square_weight + square_weight//2)
    # 초기 이미지 설정
    on_button_click_tutorial()

    
    canvas.bind_all("<KeyPress>", player_move)
    frame_buttons = tk.Frame(window)
    frame_buttons.pack()
    create_button(frame_buttons, "1단계", lambda: on_button_click(maze_make(1), end_image_path, end_coords1))
    create_button(frame_buttons, "2단계", lambda: on_button_click(maze_make(2), end_image_path, end_coords2))
    create_button(frame_buttons, "3단계", lambda: on_button_click(maze_make(3), end_image_path, end_coords3))
    create_button(frame_buttons, "듀토리얼", on_button_click_tutorial)
    create_button(frame_buttons, "카메라 켜기", on_camera_button_click)

    window.mainloop()


if __name__ == "__main__":
    # # Start the camera thread
    # camera_thread = threading.Thread(target=block_move)
    # camera_thread.start()

    # # Start the maze window in the main thread
    create_window()