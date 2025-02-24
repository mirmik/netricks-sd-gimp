#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import base64
import time
import gi

gi.require_version('Gimp', '3.0')
gi.require_version('GimpUi', '3.0')
from gi.repository import Gimp, GimpUi, Gtk, GLib, Gegl

import requests
import json
import traceback
import io
import numpy as np
import os
from PIL import Image
from enum import Enum

port = 7861
server_url = "http://127.0.0.1:" + str(port)
#TXT2IMG = "/sdapi/v1/txt2img"
#IMG2IMG = "/sdapi/v1/img2img"

config = {
    "url" : "http://127.0.0.1:" + str(port),
    "api": {
        "txt2img": ("POST", "/sdapi/v1/txt2img"),
        "img2img": ("POST", "/sdapi/v1/img2img"),
        "getModels" : ("GET", "/sdapi/v1/sd-models"),
        "getOptions" : ("GET", "/sdapi/v1/options"),
        "setOptions" : ("POST", "/sdapi/v1/options"),
    }
}

class ReturnTo(Enum):
    NEW_TAB = 1
    NEW_LAYER = 2

def run_inference(data):
    api_name = data["api_name"]
    params = data["params"]
    #print(params)

    if api_name not in config["api"]:
        raise Exception(f"API {api_name} not found in config")

    method, endpoint = config["api"][api_name]
    url = config["url"] + endpoint

    if method == "POST":
        response = requests.post(url, json=params)
    elif method == "GET":
        response = requests.get(url, params=params)
    else:
        raise Exception(f"Unknown method {method}")

    if response.status_code == 200:
        return response.json()
    else:
        print(response.text)
        raise Exception(f"Error: {response.status_code}")


def raw_image_from_base64(image_data: str):
    try:
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        image = image.convert("RGBA")
        raw_image = image.tobytes("raw", "RGBA")
        width, height = image.size
        size = (width, height)

        arr = np.frombuffer(raw_image, dtype=np.uint8).reshape((height, width, 4)).copy()
        arr[:, :, :3] = np.power(arr[:, :, :3] / 255.0, 2.2) * 255

        corrected_raw_image = arr.tobytes()

        #gimp_image = Gimp.Image.new(width, height, Gimp.ImageType.RGB_IMAGE)
        #drawable = Gimp.Layer.new(gimp_image, "Generated Image", width, height, Gimp.ImageType.RGB_IMAGE, 100, Gimp.LayerMode.NORMAL)
        #gimp_image.insert_layer(drawable, None, 0)
        
        #buffer = drawable.get_buffer()
        #rect = Gegl.Rectangle.new(0, 0, width, height)

        # Дорогой Github Copilot. 
        #Здесь не нужен stride!!!! Тут только 3 аргумента!!!
        #buffer.set(rect, "RGBA u8", raw_image)
        
        return corrected_raw_image, size
    except Exception as e:
        print(f"Error while processing image: {e}")
        traceback.print_exc()
        return None

def stable_diffusion_request(params):
    result = run_inference(params)
    if result and "images" in result:
        image_data = result["images"][0]
        return raw_image_from_base64(image_data)
    else:
        return None

class NetricsStableDiffusionDialog(Gtk.Dialog):

    def add_cfg_scale_slider(self, dialog):
        self.add_label(dialog, "Insert cfg scale:")
        scale = Gtk.Scale.new_with_range(Gtk.Orientation.HORIZONTAL, 0, 30, 0.5)
        scale.set_value(1)
        content_area = dialog.get_content_area()
        content_area.pack_start(scale, True, True, 0)
        return scale

    def add_denoising_strength_slider(self, dialog):
        self.add_label(dialog, "Insert denoising strength:")
        scale = Gtk.Scale.new_with_range(Gtk.Orientation.HORIZONTAL, 0, 1, 0.01)
        scale.set_value(0.5)
        content_area = dialog.get_content_area()
        content_area.pack_start(scale, True, True, 0)
        return scale

    def load_prefs(self):
        dialog_type = self.dialog_type
        try:
            with open(os.path.expanduser(f"~/.netricks-sd-prefs.{dialog_type}.json"), "r") as f:
                data = json.load(f)
                return data
        except Exception as e:
            print(f"Error while loading preferences: {e}")
            return None

    def on_close(self, widget):
        dialog_type = self.dialog_type
        data = self.make_prefs_object()
        print("Saving preferences: \n" + str(data))
        
        # save data to file ~/.netricks-sd-prefs.json
        with open(os.path.expanduser(f"~/.netricks-sd-prefs.{dialog_type}.json"), "w") as f:
            json.dump(data, f)


class NetricsStableDiffusionDialog_Txt2Img(NetricsStableDiffusionDialog):

    def __init__(self, title, parent, flags, image):
        self.dialog_type = "txt2img"
        data = self.load_prefs()

        super().__init__(title=title, transient_for=parent, flags=flags)
        self.set_default_size(600, 800)
        self.set_resizable(False)

        width = image.get_width()
        height = image.get_height()
        self.width_entry, self.height_entry = self.add_width_height_text_fields(self, width, height)  
        self.prompt_entry = self.add_positive_prompt_text_field(self)
        self.negative_prompt_entry = self.add_positive_prompt_text_field(self)
        self.cfg_scale_slider = self.add_cfg_scale_slider(self)
        self.steps_slider = self.add_steps_slider(self)
        self.denoising_strength_slider = self.add_denoising_strength_slider(self)

        self.apply_data(data)

        self.add_button("_OK", Gtk.ResponseType.OK)
        self.add_button("_Cancel", Gtk.ResponseType.CANCEL)

        # add action to close the dialog
        self.connect("destroy", self.on_close)


    def add_label(self, dialog, text, width=-1, height=-1):
        label = Gtk.Label()
        label.set_text(text)
        content_area = dialog.get_content_area()
        content_area.pack_start(label, True, True, 0)
        self.width = width
        self.height = height

    def add_width_height_text_fields(self, dialog, width, height):
        self.add_label(dialog, "Insert width:")
        content_area = dialog.get_content_area()
        width_entry = Gtk.Entry()
        width_entry.set_text(str(width))
        content_area.pack_start(width_entry, True, True, 0)
        self.add_label(dialog, "Insert height:")
        height_entry = Gtk.Entry()
        height_entry.set_text(str(height))
        content_area.pack_start(height_entry, True, True, 0)
        return width_entry, height_entry

    def add_positive_prompt_text_field(self, dialog):
        self.add_label(dialog, "Insert positive prompt:")
        content_area = dialog.get_content_area()
        text_entry = Gtk.Entry()
        text_entry.set_text("")
        content_area.pack_start(text_entry, True, True, 0)
        return text_entry
        
    def add_negative_prompt_text_field(self, dialog):
        self.add_label(dialog, "Insert negative prompt:")
        content_area = dialog.get_content_area()
        text_entry = Gtk.Entry()
        text_entry.set_text("")
        content_area.pack_start(text_entry, True, True, 0)
        return text

    def add_steps_slider(self, dialog):
        self.add_label(dialog, "Insert steps:")
        scale = Gtk.Scale.new_with_range(Gtk.Orientation.HORIZONTAL, 0, 100, 1)
        scale.set_value(20)
        content_area = dialog.get_content_area()
        content_area.pack_start(scale, True, True, 0)
        return scale

    def apply_prefs(self, prefs):
        print("Applying preferences")
        try:
            if prefs:
                self.width_entry.set_text(str(prefs["width"]))
                self.height_entry.set_text(str(prefs["height"]))
                self.prompt_entry.set_text(prefs["prompt"])
                self.negative_prompt_entry.set_text(prefs["negative_prompt"])
                self.cfg_scale_slider.set_value(prefs["cfg_scale"])
                self.steps_slider.set_value(prefs["steps"])
                self.denoising_strength_slider.set_value(prefs["denoising_strength"])
        except Exception as e:
            print(f"Error while applying preferences: {e}")
            traceback.print_exc()


    def make_data_object(self):
        data = {
            "api_name": "txt2img",
            "params": {
            "width": int(self.width_entry.get_text()),
            "height": int(self.height_entry.get_text()),
            "prompt": self.prompt_entry.get_text(),
            "negative_prompt": self.negative_prompt_entry.get_text(),
            "cfg_scale": self.cfg_scale_slider.get_value(),
            "steps": self.steps_slider.get_value(),
            "denoising_strength": self.denoising_strength_slider.get_value(),
            "enable_hr": False,
            "save_images" : True
            }
        }
        return data

def pil_image_from_drawable(drawable):
    width, height = drawable.get_width(), drawable.get_height()
    buffer = drawable.get_buffer()
    rect = Gegl.Rectangle.new(0, 0, width, height)
    raw_bytes = buffer.get(
        rect=rect, 
        scale=1, 
        format_name="RGBA u8",
        repeat_mode=Gegl.AbyssPolicy.CLAMP) 

    arr = np.frombuffer(raw_bytes, dtype=np.uint8).reshape((height, width, 4)).copy()
    arr[:, :, :3] = np.power(arr[:, :, :3] / 255.0, 1/2.2) * 255

    pil_img = Image.fromarray(arr.astype(np.uint8), "RGBA")
    return pil_img

def encode_gimp_image_selected_drawable_to_png(image):
    drawables = image.get_selected_drawables()
    if not drawables:
        print("Ошибка: Нет активных слоев")
        return None
    drawable = drawables[0]

    img = pil_image_from_drawable(drawable)

    # Кодируем изображение в PNG
    output_buffer = io.BytesIO()
    img.save(output_buffer, format="PNG")
    png_bytes = output_buffer.getvalue()

    return png_bytes

# def encode_gimp_image_all_selected_drawables_to_png(image):
#     drawables = image.get_selected_drawables()
#     if not drawables:
#         print("Ошибка: Нет активных слоев")
#         return None

#     width, height = image.get_width(), image.get_height()

#     img = Image.new("RGBA", (width, height))

#     for drawable in drawables:
#         drawable_img = pil_image_from_drawable(drawable)
#         offsets = drawable.get_offsets()
#         img.paste(drawable_img, offsets)

#     # Кодируем изображение в PNG
#     output_buffer = io.BytesIO()
#     img.save(output_buffer, format="PNG")
#     png_bytes = output_buffer.getvalue()

#     return png_bytes

def encode_gimp_image_as_flatten_to_pil_image(image):
    dublicate = image.duplicate()
    flatten = dublicate.flatten()
    img = pil_image_from_drawable(flatten)
    return img

def pil_image_encode_to_png(img):
    output_buffer = io.BytesIO()
    img.save(output_buffer, format="PNG")
    png_bytes = output_buffer.getvalue()
    print ("PNG size: ", len(png_bytes))
    return png_bytes

def pil_image_encode_to_base64(img):
    png_bytes = pil_image_encode_to_png(img)
    return base64.b64encode(png_bytes).decode("utf-8")

# def encode_gimp_image(gimp_image):
#     rendered = encode_gimp_image_as_flatten_to_png(gimp_image)
#     return base64.b64encode(rendered).decode("utf-8")
        


class NetricsStableDiffusionDialog_Img2Img(NetricsStableDiffusionDialog):

    def __init__(self, title, parent, flags, image):
        self.dialog_type = "img2img"
        prefs = self.load_prefs()

        super().__init__(title=title, transient_for=parent, flags=flags)
        self.set_default_size(600, 800)
        self.set_resizable(False)

        self.image = image
        self.all_image_vs_selection_chooser = self.add_all_image_vs_selection_chooser(self)
        self.prompt_entry = self.add_positive_prompt_text_field(self)
        self.negative_prompt_entry = self.add_positive_prompt_text_field(self)
        self.cfg_scale_slider = self.add_cfg_scale_slider(self)
        self.steps_slider = self.add_steps_slider(self)
        self.denoising_strength_slider = self.add_denoising_strength_slider(self)

        self.apply_data(prefs)

        self.add_button("_OK", Gtk.ResponseType.OK)
        self.add_button("_Cancel", Gtk.ResponseType.CANCEL)

        # add action to close the dialog
        self.connect("destroy", self.on_close)

    def add_all_image_vs_selection_chooser(self, dialog):
        self.add_label(dialog, "Insert image:")
        content_area = dialog.get_content_area()
        chooser = Gtk.ComboBoxText()
        chooser.append_text("All image")
        chooser.append_text("Selected drawable")
        chooser.set_active(0)
        content_area.pack_start(chooser, True, True, 0)
        return chooser

    def add_label(self, dialog, text, width=-1, height=-1):
        label = Gtk.Label()
        label.set_text(text)
        content_area = dialog.get_content_area()
        content_area.pack_start(label, True, True, 0)
        self.width = width
        self.height = height

    def add_width_height_text_fields(self, dialog, width, height):
        self.add_label(dialog, "Insert width:")
        content_area = dialog.get_content_area()
        width_entry = Gtk.Entry()
        width_entry.set_text(str(width))
        content_area.pack_start(width_entry, True, True, 0)
        self.add_label(dialog, "Insert height:")
        height_entry = Gtk.Entry()
        height_entry.set_text(str(height))
        content_area.pack_start(height_entry, True, True, 0)
        return width_entry, height_entry

    def add_positive_prompt_text_field(self, dialog):
        self.add_label(dialog, "Insert positive prompt:")
        content_area = dialog.get_content_area()
        text_entry = Gtk.Entry()
        text_entry.set_text("")
        content_area.pack_start(text_entry, True, True, 0)
        return text_entry
        
    def add_negative_prompt_text_field(self, dialog):
        self.add_label(dialog, "Insert negative prompt:")
        content_area = dialog.get_content_area()
        text_entry = Gtk.Entry()
        text_entry.set_text("")
        content_area.pack_start(text_entry, True, True, 0)
        return text


    def add_steps_slider(self, dialog):
        self.add_label(dialog, "Insert steps:")
        scale = Gtk.Scale.new_with_range(Gtk.Orientation.HORIZONTAL, 0, 100, 1)
        scale.set_value(20)
        content_area = dialog.get_content_area()
        content_area.pack_start(scale, True, True, 0)
        return scale


    def apply_data(self, data):
        print("Applying preferences")
        print(data)
        try:
            prefs = data["params"]
            if prefs:
                self.all_image_vs_selection_chooser.set_active(1 if data["selection_only"] else 0)
                self.prompt_entry.set_text(prefs["prompt"])
                self.negative_prompt_entry.set_text(prefs["negative_prompt"])
                self.steps_slider.set_value(prefs["steps"])
                self.cfg_scale_slider.set_value(prefs["cfg_scale"])
                self.denoising_strength_slider.set_value(prefs["denoising_strength"])
        except Exception as e:
            print(f"Error while applying preferences: {e}")
            traceback.print_exc()

    def selection_mode_enabled(self):
        return (self.all_image_vs_selection_chooser.get_active_text() == 
            "Selected drawable")
    
   
    def selection_rectangle(self):
        sel = self.image.get_selection()
        bounds = sel.bounds(self.image)
        return bounds
        
    def inpaiting_mask(self):
        # get selection mask image
        selection = self.image.get_selection()
        chnl = selection.save(self.image)
        buffer = chnl.get_buffer()

        # get selection mask image as PIL image
        width, height = chnl.get_width(), chnl.get_height()
        rect = Gegl.Rectangle.new(0, 0, width, height)
        raw_bytes = buffer.get(
            rect=rect,
            scale=1,
            format_name="R u8",
            repeat_mode=Gegl.AbyssPolicy.CLAMP)
            
        img = Image.frombytes("L", (width, height), raw_bytes, "raw", "L")
        return img
        

    def make_prefs_object(self):
        is_selection_only = self.selection_mode_enabled()
        data = {
            "api_name": "img2img",
            "selection_only": is_selection_only,
            "params": { 
                "prompt": self.prompt_entry.get_text(),
                "negative_prompt": self.negative_prompt_entry.get_text(),
                "cfg_scale": self.cfg_scale_slider.get_value(),
                "steps": self.steps_slider.get_value(),
                "denoising_strength": self.denoising_strength_slider.get_value(),
                "enable_hr": False,
                "inpainting_fill": 1, #original
            }
        }
        return data


    def make_data_object(self):
        data = self.make_prefs_object()
        is_selection_only = self.selection_mode_enabled()
        source = encode_gimp_image_as_flatten_to_pil_image(self.image)

        width = self.image.get_width()
        height = self.image.get_height()

        # if is_selection_only:
        #     selection = self.selection_rectangle()
        #     success, non_empty, x1, y1, x2, y2 = selection
            
        #     w = x2 - x1
        #     h = y2 - y1
        #     x = x1
        #     y = y1

        #     width = w
        #     height = h
            
        #     # get part of the source
        #     source = source.crop((x, y, x + w, y + h))

        inpaiting_mask = self.inpaiting_mask()

        source_encoded = pil_image_encode_to_base64(source)
        mask_encoded = pil_image_encode_to_base64(inpaiting_mask)

        data["params"]["width"] = width
        data["params"]["height"] = height
        data["params"]["save_images"] = True
        data["params"]["init_images"] = [source_encoded]
        data["params"]["mask"] = mask_encoded
        return data







class NetricsStableDiffusionPlugin(Gimp.PlugIn):
    def do_query_procedures(self):
        print("Querying procedures")
        arr = [
            "txt2img",
            "img2img",
            "clean"
            ]
        return arr

    def do_set_i18n(self, name):
        return False

    def name_to_action(self, name):
        if name == "txt2img":
            return self.run_txt2img
        elif name == "img2img":
            return self.run_img2img
        elif name == "clean":
            return self.run_clean
        else:
            return None

    def name_to_label(self, name):
        if name == "txt2img":
            return "Generate Image from Text"
        elif name == "img2img":
            return "Generate Image from Image"
        elif name == "clean":
            return "Clean NGen. layers"
        else:
            return None

    def do_create_procedure(self, name):
        print("Creating procedure for", name)
        procedure = Gimp.ImageProcedure.new(self, name,
                                            Gimp.PDBProcType.PLUGIN,
                                            self.name_to_action(name), None)

        procedure.set_image_types("*")

        procedure.set_menu_label(self.name_to_label(name))
        procedure.add_menu_path('<Image>/NetricksSD')

        procedure.set_documentation("Generates an image from text using Stable Diffusion API", "Netricks", "2025")
        procedure.set_attribution("netricks", "netricks", "2025")

        return procedure


    def run_txt2img(self, procedure, run_mode, image, drawables, config, run_data):
        dialog = NetricsStableDiffusionDialog_Txt2Img(
            "Netricks Stable Diffusion", None, 0, image)
        dialog.show_all()
        response = dialog.run()

        data = dialog.make_data_object()
        dialog.destroy()

        if response == Gtk.ResponseType.CANCEL:
            return procedure.new_return_values(Gimp.PDBStatusType.CANCEL, GLib.Error())

        return self.doit(data, procedure, image, ReturnTo.NEW_TAB)

    def run_img2img(self, procedure, run_mode, image, drawables, config, run_data):
        dialog = NetricsStableDiffusionDialog_Img2Img(
            "Netricks Stable Diffusion", None, 0, image)
        dialog.show_all()
        response = dialog.run()

        data = dialog.make_data_object()

        if data is None:
            raise Exception("data is None")
        dialog.destroy()

        if response == Gtk.ResponseType.CANCEL:
            return procedure.new_return_values(Gimp.PDBStatusType.CANCEL, GLib.Error())

        return self.doit(data, procedure, image, ReturnTo.NEW_TAB)

    def run_clean(self, procedure, run_mode, image, drawables, config, run_data):
        layers = image.get_layers()
        for layer in layers:
            if layer.get_name().startswith("NGen."):
                image.remove_layer(layer)

        return procedure.new_return_values(Gimp.PDBStatusType.SUCCESS, GLib.Error())

    def doit(self, params, procedure, image, mode):
        if params is None:
            raise Exception("params is None")

        width = params["params"]["width"]
        height = params["params"]["height"]
        mode = ReturnTo.NEW_LAYER
        if mode == ReturnTo.NEW_TAB:
            output_image = Gimp.Image.new(width, height, Gimp.ImageType.RGB_IMAGE)
            Gimp.Display.new(output_image)
        else:
            output_image = image

        # Отправляем запрос к API
        raw_image, size = stable_diffusion_request(params)
        width, height = size

        if raw_image:
            layer = Gimp.Layer.new(output_image, "NGen.Image", width, height, Gimp.ImageType.RGB_IMAGE, 100, Gimp.LayerMode.NORMAL)
            output_image.insert_layer(layer, None, 0)
            buffer = layer.get_buffer()
            rect = Gegl.Rectangle.new(0, 0, width, height)
            buffer.set(rect, "RGBA u8", raw_image)

            if mode == ReturnTo.NEW_TAB:
                original_layer = Gimp.Layer.new(output_image, "Original Image", width, height, Gimp.ImageType.RGB_IMAGE, 100, Gimp.LayerMode.NORMAL)
                output_image.insert_layer(original_layer, None, 1)
                buffer = original_layer.get_buffer()
                rect = Gegl.Rectangle.new(0, 0, width, height)
                flattened = image.dublicate().flatten()
                raw_bytes = flattened.get_buffer().get(
                    rect=rect, 
                    scale=1, 
                    format_name="RGBA u8",
                    repeat_mode=Gegl.AbyssPolicy.CLAMP)
                buffer.set(rect, "RGBA u8", raw_bytes)

            return procedure.new_return_values(Gimp.PDBStatusType.SUCCESS, GLib.Error())
        else:
            return procedure.new_return_values(Gimp.PDBStatusType.CANCEL, GLib.Error())


Gimp.main(NetricsStableDiffusionPlugin.__gtype__, sys.argv)