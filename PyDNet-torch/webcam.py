import tkinter as tk
from PIL import Image, ImageTk
from ffpyplayer.player import MediaPlayer


class CameraApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Video della fotocamera")

        # Configura il display della fotocamera
        self.label = tk.Label(root)
        self.label.pack()

        # Avvia la fotocamera
        self.video_stream = MediaPlayer("0", ff_opts={"format": "avfoundation"})

        # Aggiorna il frame della fotocamera
        self.update_frame()

    def update_frame(self):
        frame, val = self.video_stream.get_frame()
        if val != "eof" and frame is not None:
            img, t = frame
            image = Image.fromarray(img.to_bytearray()[0])

            # Converti l'immagine per tkinter
            image_tk = ImageTk.PhotoImage(image)

            # Aggiorna l'immagine sulla label
            self.label.config(image=image_tk)
            self.label.image = image_tk

        # Richiama la funzione per aggiornare il frame
        self.root.after(10, self.update_frame)

    def __del__(self):
        # Chiudi il video stream
        self.video_stream.close_player()


root = tk.Tk()
app = CameraApp(root)
root.mainloop()
