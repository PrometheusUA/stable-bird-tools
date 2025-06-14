import os
import torch
import pandas as pd
import ast


df = pd.read_csv('/home/jovyan/data/datasets/birdset_m_denoised/train.csv')
df['basename'] = df['filepath'].apply(os.path.basename)
df = df.drop_duplicates(subset='basename')
df = df.set_index('basename')
df['ebird_code_secondary'] = df['ebird_code_secondary'].apply(ast.literal_eval)

birdset_ID2CLASS = ["gretit1", "eurbla", "comcha", "comchi1", "eurrob1", "sonthr1", "blackc1", "blutit", "redcro", "winwre4", "coatit2", "houspa", "grswoo", "comnig1", "dunnoc1", "skylar", "comrav", "eurgre1", "eurgol", "trepip", "eurnut2", "eurjay1", "houwre", "eursta", "cetwar1", "barswa", "eurlin1", "lottit1", "carcro1", "martit2", "eurbul", "sarwar1", "eugori2", "spofly1", "sonspa", "shttre1", "firecr1", "eurmag1", "eurser1", "mallar3", "woolar1", "cretit2", "eucdov", "zitcis1", "spotow", "bcnher", "cirbun1", "bewwre", "rewbla", "rucspa1", "norcar", "combuz1", "grywag", "cowpig1", "amerob", "melwar1", "swathr", "comyel", "grekis", "daejun", "carwre", "warvir", "eurwry", "roahaw", "cangoo", "banana", "gretin1", "cintin1", "littin1", "undtin1", "bartin2", "horscr1", "hawgoo", "snogoo", "tunswa", "wooduc", "grhcha1", "specha3", "colcha1", "spigua1", "mouqua", "calqua", "stwqua1", "wiltur", "kalphe", "blkfra", "chukar", "ercfra", "compau", "compot1", "annhum", "buvhum1", "stvhum2", "rtlhum", "andeme1", "strcuc1", "squcuc1", "yebcuc", "scapig2", "batpig1", "pavpig2", "rebpig1", "plupig2", "rudpig", "eutdov", "blgdov1", "ruqdov", "whtdov", "grfdov1", "moudov", "gycwor1", "amgplo", "killde", "amewoo", "sposan", "solsan", "ribgul", "barpet", "hawpet1", "greibi1", "grbher3", "coohaw", "gryhaw2", "reshaw", "hawhaw", "amapyo1", "fepowl", "trsowl", "tabsco1", "brdowl", "blttro1", "gnbtro1", "viotro3", "blctro1", "coltro1", "garkin1", "rinkin1", "belkin1", "bucmot2", "bucmot4", "higmot1", "blfjac1", "wespuf1", "blfnun1", "gilbar1", "letbar1", "rehbar1", "kebtou1", "whttou1", "acowoo", "yetwoo2", "hofwoo1", "rebwoo", "wilsap", "yebsap", "dowwoo", "litwoo2", "haiwoo", "whhwoo", "whtwoo2", "gogwoo1", "norfli", "scbwoo5", "rinwoo1", "linwoo1", "pilwoo", "renwoo1", "blacar1", "yehcar1", "laufal1", "baffal1", "coffal1", "buffal1", "orcpar", "cowpar1", "blhpar1", "whcpar", "brwpar1", "whfpar1", "meapar", "orfpar", "duhpar", "rebmac2", "crfpar", "oliwoo1", "plbwoo1", "citwoo1", "lobwoo1", "amabaw1", "strwoo2", "elewoo1", "butwoo1", "stbwoo2", "sthwoo1", "strxen1", "crfgle1", "chwfog1", "btfgle1", "azaspi1", "spwant2", "bltant2", "pygant1", "whfant2", "lowant1", "gryant1", "pltant1", "dutant2", "barant1", "plwant1", "fasant1", "greant1", "bsbeye1", "gryant2", "pluant1", "goeant1", "rucant2", "blfant1", "rufant3", "astgna1", "wibpip1", "yectyr1", "forela1", "yebela1", "whltyr1", "rinant2", "goftyr1", "whbtot1", "ruftof1", "cotfly1", "yemfly1", "gycfly1", "gocspa1", "whcspa1", "eulfly1", "easpho", "olsfly", "wewpew", "eawpew", "aldfly", "hamfly", "dusfly", "pasfly", "pirfly1", "rumfly1", "socfly1", "grcfly1", "bobfly1", "trokin", "easkin", "gramou1", "whrsir1", "ducfly", "grcfly", "ducatt1", "brratt1", "putfru1", "scrpih1", "blfcot1", "lotman1", "batman1", "royfly1", "mastit1", "cinmou1", "whwbec1", "blcbec1", "rotbec", "ducgre1", "reevir1", "hutvir", "yetvir", "buhvir", "casvir", "elepai", "blcjay1", "grnjay", "brnjay", "blujay", "stejay", "clanut", "amegfi", "pinsis", "puteup1", "yeceup1", "yeteup1", "blbthr1", "whnrob1", "hauthr1", "clcrob", "bubwre1", "muswre2", "melbla1", "rusbla", "brebla", "comgra", "grtgra", "ovenbi1", "louwat", "norwat", "buwwar", "bawwar", "tenwar", "orcwar", "naswar", "macwar", "kenwar", "hoowar", "amered", "babwar", "bkbwar", "yelwar", "chswar", "yerwar", "btywar", "towwar", "herwar", "thelar1", "crelar1", "btnwar", "rucwar1", "rucwar", "wlswar", "sltred", "scatan", "westan", "rcatan1", "robgro", "bkhgro", "bubgro2", "lazbun", "bugtan", "blctan1", "scrtan1", "partan1", "woothr", "obnthr1", "herthr", "veery", "evegro", "strsal1", "sibtan2", "buggna", "whbnut", "rebnut", "brncre", "grycat", "rocpet1", "comwax", "easwar1", "darwar1", "tuftit", "chbchi", "bkcchi", "mouchi", "amepip", "gcrfin", "palila", "iiwi", "apapan", "hawcre", "akepa1", "hawama", "purfin", "casfin", "houfin", "wegspa1", "pregrs2", "gnttow", "eastow", "boboli", "ruboro1", "olioro1", "monoro1", "sobcac1", "yercac1", "yebori1", "balori", "bnhcow", "ruckin", "gockin", "runwre1", "thlwre1", "rocwre", "whiwre1", "rubwre1", "rawwre1", "plawre1", "moublu", "easblu", "towsol", "omao", "andsol1", "warwhe1", "ccbfin", "foxspa", "whcspa", "whtspa", "vesspa", "linspa", "swaspa", "treswa", "jabwar", "grasal3", "butsal1", "blhsal1", "yefgra1", "flrtan1", "amecro", "cedwax", "yefcan", "reblei", "melthr"]
birdset_CLASS2ID = {v: i for i, v in enumerate(birdset_ID2CLASS)}


def get_custom_metadata(info, audio):
    filename = os.path.basename(info["relpath"])
    
    try:
        row = df.loc[filename]
    except KeyError:
        raise ValueError(f"No metadata found for file: {filename}")

    main_class = row['ebird_code']
    other_classnames = row['ebird_code_secondary']

    main_id = birdset_CLASS2ID.get(main_class)
    secondary_ids = [
        birdset_CLASS2ID[c] for c in other_classnames if c in birdset_CLASS2ID
    ]

    return {"class": (main_id, secondary_ids)}
