import cv2
import os
import glob
import argparse
import numpy as np


def load_class_names(names_path):
	if not names_path or not os.path.isfile(names_path):
		return []
	with open(names_path, 'r', encoding='utf-8') as f:
		lines = [l.strip() for l in f.readlines() if l.strip()]
	return lines


def compute_label_dir(image_dir):
	base = os.path.basename(os.path.normpath(image_dir))
	parent = os.path.dirname(os.path.normpath(image_dir))
	return os.path.join(parent, base + '_label')


def get_label_txt_path_for_image(image_path, image_dir):
	# prefer label in label_dir mirroring relative structure, else fallback to same-folder txt
	label_dir = compute_label_dir(image_dir)
	try:
		rel = os.path.relpath(image_path, image_dir)
	except Exception:
		rel = os.path.basename(image_path)
	rel_noext = os.path.splitext(rel)[0] + '.txt'
	txt = os.path.join(label_dir, rel_noext)
	if os.path.exists(txt):
		return txt
	local = os.path.splitext(image_path)[0] + '.txt'
	return local


def save_labels_to_labeldir(image_path, image_dir, boxes, img_w, img_h):
	label_dir = compute_label_dir(image_dir)
	try:
		rel = os.path.relpath(image_path, image_dir)
	except Exception:
		rel = os.path.basename(image_path)
	rel_noext = os.path.splitext(rel)[0] + '.txt'
	out_path = os.path.join(label_dir, rel_noext)
	os.makedirs(os.path.dirname(out_path), exist_ok=True)
	save_yolo_labels(out_path, boxes, img_w, img_h)
	return out_path


def yolo_to_box(xc, yc, w, h, img_w, img_h):
	cx = xc * img_w
	cy = yc * img_h
	wbox = w * img_w
	hbox = h * img_h
	x1 = int(cx - wbox / 2)
	y1 = int(cy - hbox / 2)
	x2 = int(cx + wbox / 2)
	y2 = int(cy + hbox / 2)
	return x1, y1, x2, y2


def box_to_yolo(x1, y1, x2, y2, img_w, img_h):
	x1, x2 = min(x1, x2), max(x1, x2)
	y1, y2 = min(y1, y2), max(y1, y2)
	w = (x2 - x1) / img_w
	h = (y2 - y1) / img_h
	xc = (x1 + x2) / 2.0 / img_w
	yc = (y1 + y2) / 2.0 / img_h
	return xc, yc, w, h


def read_yolo_labels(txt_path, img_w, img_h):
	boxes = []
	if not os.path.isfile(txt_path):
		return boxes
	with open(txt_path, 'r', encoding='utf-8') as f:
		for line in f:
			parts = line.strip().split()
			if len(parts) < 5:
				continue
			cls = int(float(parts[0]))
			xc, yc, w, h = map(float, parts[1:5])
			x1, y1, x2, y2 = yolo_to_box(xc, yc, w, h, img_w, img_h)
			boxes.append({'class': cls, 'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2})
	return boxes


def save_yolo_labels(txt_path, boxes, img_w, img_h):
	lines = []
	for b in boxes:
		xc, yc, w, h = box_to_yolo(b['x1'], b['y1'], b['x2'], b['y2'], img_w, img_h)
		lines.append(f"{b['class']} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
	with open(txt_path, 'w', encoding='utf-8') as f:
		f.writelines(lines)


class Annotator:
	def __init__(self, image_dir, names_path=None, exts=('jpg','jpeg','png','bmp')):
		self.image_dir = image_dir
		self.names = load_class_names(names_path)
		self.class_id = 0
		self.exts = exts
		self.images = self._find_images()
		self.idx = 0
		self.boxes = []
		self.drawing = False
		self.start_pt = (0,0)
		self.current_rect = None
		self.window_name = 'YOLO Annotator'

	def _find_images(self):
		imgs = []
		for e in self.exts:
			imgs.extend(glob.glob(os.path.join(self.image_dir, f'**/*.{e}'), recursive=True))
		imgs = sorted(imgs)
		return imgs

	def load_image(self):
		if not self.images:
			return None
		path = self.images[self.idx]
		img = cv2.imread(path)
		if img is None:
			return None
		h, w = img.shape[:2]
		# read labels preferentially from sibling <folder>_label directory
		txt_path = get_label_txt_path_for_image(path, self.image_dir)
		self.boxes = read_yolo_labels(txt_path, w, h)
		return img, path

	def draw(self, img):
		canvas = img.copy()
		for b in self.boxes:
			color = (0, 255, 0)
			cv2.rectangle(canvas, (b['x1'], b['y1']), (b['x2'], b['y2']), color, 2)
			label = str(b['class'])
			if 0 <= b['class'] < len(self.names):
				label = f"{b['class']}: {self.names[b['class']]}"
			cv2.putText(canvas, label, (b['x1'], max(b['y1']-6,10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
		# draw current
		if self.current_rect is not None:
			x1,y1,x2,y2 = self.current_rect
			cv2.rectangle(canvas, (x1,y1), (x2,y2), (0,0,255), 1)
		# info text
		info = f"{self.idx+1}/{len(self.images)} cls={self.class_id} ({len(self.names)} names)"
		cv2.putText(canvas, info, (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
		return canvas

	def on_mouse(self, event, x, y, flags):
		if event == cv2.EVENT_LBUTTONDOWN:
			self.drawing = True
			self.start_pt = (x,y)
			self.current_rect = (x,y,x,y)
		elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
			x0,y0 = self.start_pt
			self.current_rect = (x0,y0,x,y)
		elif event == cv2.EVENT_LBUTTONUP and self.drawing:
			self.drawing = False
			x0,y0 = self.start_pt
			rect = (x0,y0,x,y)
			x1, y1, x2, y2 = rect
			if abs(x2-x1) > 5 and abs(y2-y1) > 5:
				self.boxes.append({'class': self.class_id, 'x1': min(x1,x2), 'y1': min(y1,y2), 'x2': max(x1,x2), 'y2': max(y1,y2)})
			self.current_rect = None
		elif event == cv2.EVENT_RBUTTONDOWN:
			# delete box under cursor
			for i in range(len(self.boxes)-1, -1, -1):
				b = self.boxes[i]
				if x >= b['x1'] and x <= b['x2'] and y >= b['y1'] and y <= b['y2']:
					self.boxes.pop(i)
					break

	def save_current(self, image_path, img_w, img_h):
		out = save_labels_to_labeldir(image_path, self.image_dir, self.boxes, img_w, img_h)
		return out

	def run(self):
		if not self.images:
			print('No images found in', self.image_dir)
			return
		cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
		cv2.resizeWindow(self.window_name, 1200, 800)
		while True:
			data = self.load_image()
			if data is None:
				print('Failed loading image', self.images[self.idx])
				return
			img, path = data
			h, w = img.shape[:2]
			cv2.setMouseCallback(self.window_name, lambda e,x,y,f,p=self: p.on_mouse(e,x,y,f))
			while True:
				canvas = self.draw(img)
				cv2.imshow(self.window_name, canvas)
				key = cv2.waitKey(20) & 0xFF
				if key == 255:
					continue
				if key == ord('q'):
					# save then quit
					self.save_current(path, w, h)
					cv2.destroyAllWindows()
					return
				elif key == ord('n') or key == ord('d'):
					# next image
					self.save_current(path, w, h)
					self.idx = min(self.idx + 1, len(self.images)-1)
					break
				elif key == ord('p'):
					self.save_current(path, w, h)
					self.idx = max(self.idx - 1, 0)
					break
				elif key == ord('s'):
					self.save_current(path, w, h)
					print('Saved', os.path.splitext(path)[0] + '.txt')
				elif key == ord('z'):
					if self.boxes:
						self.boxes.pop()
				elif key == ord('c'):
					# cycle class
					if self.names:
						self.class_id = (self.class_id + 1) % len(self.names)
				elif ord('0') <= key <= ord('9'):
					num = key - ord('0')
					# if within name range, set; otherwise set value
					if num < len(self.names):
						self.class_id = num
					else:
						self.class_id = num
				elif key == ord('h'):
					print_help(self.names)


def print_help(names):
	print('\nYOLO Annotator Controls:')
	print('  Left-drag: draw box')
	print('  Right-click: delete box under cursor')
	print('  n / d: next image (saves current)')
	print('  p: previous image (saves current)')
	print('  s: save labels for current image')
	print('  z: undo last box')
	print('  c: cycle class id')
	print('  0-9: set class id (if >9 classes, use c to cycle)')
	print('  q: save and quit')
	if names:
		print('\nClasses:')
		for i, n in enumerate(names):
			print(f'  {i}: {n}')


def main():
	parser = argparse.ArgumentParser(description='Simple YOLO format image annotator')
	parser.add_argument('--images', '-i', default='.', help='images folder (recursive search)')
	parser.add_argument('--names', '-n', default=os.path.join(os.getcwd(), 'names.txt'), help='path to names.txt')
	parser.add_argument('--gui', action='store_true', help='use tkinter GUI')
	args = parser.parse_args()

	if args.gui:
		try:
			import tkinter as tk
			from tkinter import ttk, messagebox, filedialog
			from PIL import Image, ImageTk
		except Exception as e:
			print('Tkinter/Pillow not available. Install pillow and ensure tkinter is present.')
			print('Error:', e)
			return

		class TKAnnotator:
			def __init__(self, image_dir, names_path=None, exts=('jpg','jpeg','png','bmp')):
				self.image_dir = image_dir
				self.names = load_class_names(names_path)
				self.exts = exts
				self.images = []
				for e in self.exts:
					self.images.extend(glob.glob(os.path.join(self.image_dir, f'**/*.{e}'), recursive=True))
				self.images = sorted(self.images)
				self.idx = 0
				self.boxes = []
				self.class_id = 0
				self.scale = 1.0
				self.current_rect_id = None
				self.current_rect = None

				self.root = tk.Tk()
				self.root.title('YOLO Annotator GUI')

				control_frame = tk.Frame(self.root)
				control_frame.pack(side=tk.TOP, fill=tk.X)

				# folder entry + browse
				self.folder_var = tk.StringVar(value=self.image_dir)
				folder_entry = tk.Entry(control_frame, textvariable=self.folder_var, width=40)
				folder_entry.pack(side=tk.LEFT, padx=4)
				btn_browse = tk.Button(control_frame, text='Browse', command=self.browse)
				btn_browse.pack(side=tk.LEFT, padx=4)

				# class dropdown or spinbox
				if self.names:
					self.class_var = tk.StringVar(value=self.names[0])
					self.dropdown = ttk.Combobox(control_frame, textvariable=self.class_var, values=self.names, state='readonly')
					self.dropdown.pack(side=tk.LEFT, padx=4)
					self.dropdown.bind('<<ComboboxSelected>>', self.on_dropdown)
				else:
					self.class_var = tk.StringVar(value='0')

				self.spin = tk.Spinbox(control_frame, from_=0, to=999, width=5, command=self.on_spin)
				self.spin.pack(side=tk.LEFT, padx=4)

				btn_save = tk.Button(control_frame, text='Save', command=self.save)
				btn_save.pack(side=tk.LEFT, padx=4)
				btn_prev = tk.Button(control_frame, text='Prev', command=self.prev)
				btn_prev.pack(side=tk.LEFT, padx=4)
				btn_next = tk.Button(control_frame, text='Next', command=self.next)
				btn_next.pack(side=tk.LEFT, padx=4)
				btn_undo = tk.Button(control_frame, text='Undo', command=self.undo)
				btn_undo.pack(side=tk.LEFT, padx=4)
				btn_quit = tk.Button(control_frame, text='Quit', command=self.quit)
				btn_quit.pack(side=tk.LEFT, padx=4)

				self.canvas = tk.Canvas(self.root, cursor='cross')
				self.canvas.pack(side=tk.TOP, expand=True, fill=tk.BOTH)
				self.canvas.bind('<ButtonPress-1>', self.on_button_press)
				self.canvas.bind('<B1-Motion>', self.on_move)
				self.canvas.bind('<ButtonRelease-1>', self.on_button_release)

				self.img_on_canvas = None
				if not self.images:
					messagebox.showinfo('Info', f'No images found in {self.image_dir}')
					self.root.destroy()
					return
				self.load_image()
				self.root.mainloop()

			def load_image(self):
				path = self.images[self.idx]
				pil = Image.open(path).convert('RGB')
				iw, ih = pil.size
				max_w, max_h = 1200, 800
				self.scale = min(1.0, max_w / iw, max_h / ih)
				disp_w, disp_h = int(iw * self.scale), int(ih * self.scale)
				self.display_image = pil.resize((disp_w, disp_h), Image.LANCZOS)
				self.photo = ImageTk.PhotoImage(self.display_image)
				self.canvas.config(width=disp_w, height=disp_h)
				self.canvas.delete('all')
				self.img_on_canvas = self.canvas.create_image(0, 0, anchor='nw', image=self.photo)
				# load labels
				# read labels preferentially from sibling <folder>_label directory
				txt_path = get_label_txt_path_for_image(path, self.image_dir)
				self.boxes = read_yolo_labels(txt_path, iw, ih)
				# draw boxes
				for b in self.boxes:
					self._draw_box_on_canvas(b)

			def browse(self):
				folder = filedialog.askdirectory(initialdir=self.image_dir)
				if not folder:
					return
				self.image_dir = folder
				self.folder_var.set(folder)
				# rebuild image list
				imgs = []
				for e in self.exts:
					imgs.extend(glob.glob(os.path.join(self.image_dir, f'**/*.{e}'), recursive=True))
				imgs = sorted(imgs)
				self.images = imgs
				self.idx = 0
				if not self.images:
					messagebox.showinfo('Info', f'No images found in {self.image_dir}')
					self.canvas.delete('all')
					return
				self.load_image()

			def _draw_box_on_canvas(self, b):
				x1 = int(b['x1'] * self.scale) if isinstance(b['x1'], float) else int(b['x1']*self.scale)
				y1 = int(b['y1'] * self.scale) if isinstance(b['y1'], float) else int(b['y1']*self.scale)
				x2 = int(b['x2'] * self.scale) if isinstance(b['x2'], float) else int(b['x2']*self.scale)
				y2 = int(b['y2'] * self.scale) if isinstance(b['y2'], float) else int(b['y2']*self.scale)
				color = 'green'
				tag = f"box{len(self.canvas.find_withtag('all'))}"
				self.canvas.create_rectangle(x1, y1, x2, y2, outline=color, width=2, tags=('box', tag))
				lbl = str(b['class'])
				if 0 <= b['class'] < len(self.names):
					lbl = f"{b['class']}: {self.names[b['class']]}"
				self.canvas.create_text(x1+4, max(y1-6,8), text=lbl, anchor='nw', fill=color, tags=('box',))

			def on_dropdown(self, event=None):
				if self.names:
					sel = self.class_var.get()
					if sel in self.names:
						self.class_id = self.names.index(sel)
						self.spin.delete(0, 'end')
						self.spin.insert(0, str(self.class_id))

			def on_spin(self):
				try:
					v = int(self.spin.get())
					self.class_id = v
					if 0 <= v < len(self.names):
						self.class_var.set(self.names[v])
				except Exception:
					pass

			def on_button_press(self, event):
				self.start_x = event.x
				self.start_y = event.y
				self.current_rect_id = self.canvas.create_rectangle(self.start_x, self.start_y, event.x, event.y, outline='red', width=1)

			def on_move(self, event):
				if self.current_rect_id:
					self.canvas.coords(self.current_rect_id, self.start_x, self.start_y, event.x, event.y)

			def on_button_release(self, event):
				if not self.current_rect_id:
					return
				x1, y1, x2, y2 = self.canvas.coords(self.current_rect_id)
				self.canvas.delete(self.current_rect_id)
				self.current_rect_id = None
				# convert back to original image coords
				ox1 = int(round(x1 / self.scale))
				oy1 = int(round(y1 / self.scale))
				ox2 = int(round(x2 / self.scale))
				oy2 = int(round(y2 / self.scale))
				if abs(ox2-ox1) > 5 and abs(oy2-oy1) > 5:
					b = {'class': self.class_id, 'x1': min(ox1,ox2), 'y1': min(oy1,oy2), 'x2': max(ox1,ox2), 'y2': max(oy1,oy2)}
					self.boxes.append(b)
					self._draw_box_on_canvas(b)

			def save(self):
				path = self.images[self.idx]
				pil = Image.open(path)
				iw, ih = pil.size
				out = save_labels_to_labeldir(path, self.image_dir, self.boxes, iw, ih)
				tk.messagebox.showinfo('Saved', f'Saved {out}')

			def prev(self):
				if self.idx > 0:
					# save then go
					self.save()
					self.idx -= 1
					self.load_image()

			def next(self):
				if self.idx < len(self.images)-1:
					self.save()
					self.idx += 1
					self.load_image()

			def undo(self):
				if self.boxes:
					self.boxes.pop()
					# redraw
					self.canvas.delete('all')
					self.canvas.create_image(0, 0, anchor='nw', image=self.photo)
					for b in self.boxes:
						self._draw_box_on_canvas(b)

			def quit(self):
				self.save()
				self.root.destroy()

		TKAnnotator(args.images, args.names)
	else:
		ann = Annotator(args.images, args.names)
		print_help(ann.names)
		ann.run()


if __name__ == '__main__':
	main()
