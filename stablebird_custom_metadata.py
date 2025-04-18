import os


def get_custom_metadata(info, audio):
    bird_species = os.path.basename(info["relpath"])

    # Pass in the relative path of the audio file as the prompt
    return {"prompt": f"Bird vocalization of {bird_species}"}
