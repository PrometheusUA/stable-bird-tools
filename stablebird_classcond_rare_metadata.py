import os

rare_ID2CLASS = ['blchaw1', 'bobher1', 'neocor', 'palhor2', 'bicwre1', 'brtpar1', 'olipic1', 'yelori1', 'piwtyr1', 'grepot1', 'crbtan1', 'labter1', 'shghum1', 'rufmot1', 'sahpar1', 'woosto', 'royfly1', 'bubcur1', 'rutpuf1', 'whmtyr1', 'amakin1', 'rubsee1', 'plctan1', 'cregua1', 'yectyr1', 'blkvul', 'yehbla2', 'recwoo1', 'grasal4', 'spepar1', 'cinbec1', 'fotfly', 'ampkin1', 'piepuf1', 'crebob1', 'shtfly1', 'bucmot3', 'blctit1', 'plukit1', 'cocher1', 'cargra1', 'tbsfin1', 'anhing', 'rebbla1', 'whwswa1', 'thlsch3', 'spbwoo1', 'verfly', 'rosspo1', 'savhaw1', 'ruther1', 'grysee1', 'turvul', 'norscr1', 'bafibi1', 'gretin1', 'colara1', 'ragmac1', 'whttro1']

rare_CLASS2ID = {v: k for k, v in enumerate(rare_ID2CLASS)}


def get_custom_metadata(info, audio):
    bird_species = os.path.basename(os.path.dirname(info["relpath"]))

    # Pass in the relative path of the audio file as the prompt
    return {"class": rare_CLASS2ID[bird_species]}
