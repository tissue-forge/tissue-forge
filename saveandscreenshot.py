import tissue_forge as tf
import ipywidgets as widgets
from IPython.display import display
from ipyfilechooser import FileChooser
import os

save_location_set = False
screenshot_location_set = False

def save_widget():
    def find_newest_filepath(_save_folder):
        file_extension = ".json"
        filepath = _save_folder + file_extension
        count = 1
        while os.path.exists(filepath):
            filepath = f"{_save_folder}({count}){file_extension}"
            count += 1
        return filepath
    
    def _handle_folder_selection(save_during: bool):
        def _inner(fdialog):
            global save_folder, save_location_set
            save_folder = fdialog.selected
            print(f'Selected folder: {save_folder}')
            
            if save_folder:
                save_location_set = True

                if save_during:
                    save_filepath = find_newest_filepath(save_folder)

                    print('1 Trying filepath:', save_filepath)

                    result = tf.io.toFile(save_filepath)
                    if result == 0:
                        print("Saved successfully.")
                    else:
                        print("Failed to save.")
                    print(f"Save location set: {save_filepath}")

        return _inner
    
    def save_render_widgets(output):
        output.clear_output()
        saveButton=widgets.Button(
                    description='Save simulation',
                    disabled=False,
                    button_style='info',
                    tooltip='Save a simulation to file',
                    clear_output=True)

        def save_button_with_dialog():
            folder_chooser = FileChooser()
            folder_chooser.title = 'Select a folder'
            folder_chooser.default_path = '.'
            folder_chooser.show_only_dirs = False
            folder_chooser.default_filename = 'TissueForge_SavedSimulation'

            folder_chooser.register_callback(_handle_folder_selection(True))

            output2.clear_output()
            with output2:
                display(folder_chooser)
        
        def save_button(*args, **kwargs):
            global save_location_set, save_folder
            
            if not save_location_set:
                save_button_with_dialog()
                
            else:
                save_filepath = find_newest_filepath(save_folder)

                print('2 Trying filepath:', save_filepath)

                result = tf.io.toFile(save_filepath)
                if result == 0:
                    print("Saved successfully.")
                else:
                    print("Failed to save.")
                
        saveButton.on_click(save_button)

        hbox_save = widgets.HBox([saveButton])
        output2 = widgets.Output()
        vbox_all = widgets.VBox([hbox_save, output2])

        with output:
            display(vbox_all)
        
        return saveButton, output2
    
    output = widgets.Output()
    display(output)
    _ = save_render_widgets(output)

def screenshot_widget():
    def find_newest_screenshot_filepath(_screenshot_folder):
        file_extension = ".png"
        filepath = _screenshot_folder + file_extension
        count = 1
        while os.path.exists(filepath):
            filepath = f"{_screenshot_folder}({count}){file_extension}"
            count += 1
        return filepath

    def _handle_screenshot_folder_selection(screenshot_during: bool):
        def _s_inner(fdialog):
            global screenshot_folder, screenshot_location_set
            screenshot_folder = fdialog.selected
            print(f'Selected folder for screenshot: {screenshot_folder}')
            
            if screenshot_folder:
                screenshot_location_set = True

                if screenshot_during:
                    screenshot_filepath = find_newest_screenshot_filepath(screenshot_folder)

                    print('1 Trying filepath:', screenshot_filepath)

                    decorate = False
                    bgcolor = [0.0,0.0,0.0]
                    result = tf.system.screenshot(screenshot_filepath, decorate, bgcolor)
                    if result == 0:
                        print("Screenshot saved successfully.")
                    else:
                        print("Failed to save.")
                    print(f"Screenshot location set: {screenshot_filepath}")

        return _s_inner
    
    def screenshot_render_widgets(output):
        output.clear_output()
        screenshotButton=widgets.Button(
                description='Screenshot simulation',
                disabled=False,
                button_style='info',
                tooltip='Save a screenshot of the current render window to file',
                clear_output=True)
        def screenshot_button_with_dialog():
            folder_chooser = FileChooser()
            folder_chooser.title = 'Select a folder'
            folder_chooser.default_path = '.'
            folder_chooser.show_only_dirs = False
            folder_chooser.default_filename = 'TissueForge_ScreenshotSimulation'


            folder_chooser.register_callback(_handle_screenshot_folder_selection(True))

            output2.clear_output()
            with output2:
                display(folder_chooser)
        
        def screenshot_button(*args, **kwargs):
            global screenshot_location_set, screenshot_folder
            
            if not screenshot_location_set:
                screenshot_button_with_dialog()
            else:
                screenshot_filepath = find_newest_screenshot_filepath(screenshot_folder)

                print('2 Trying filepath:', screenshot_filepath)
                decorate = False
                bgcolor = [0.0,0.0,0.0]
                result = tf.system.screenshot(screenshot_filepath, decorate, bgcolor)
                if result == 0:
                    print("Screenshot saved successfully.")
                else:
                    print("Failed to screenshot.")
        
        screenshotButton.on_click(screenshot_button)
        hbox_Screenshot = widgets.HBox([screenshotButton])
        output2 = widgets.Output()
        vbox_all = widgets.VBox([hbox_Screenshot, output2])
        with output:
            display(vbox_all)
        return screenshotButton, output2
    output = widgets.Output()

    display(output)
    _ = screenshot_render_widgets(output)

