"""Apply a curated artist -> genre map to paper/benchmark_candidates.csv."""
import csv
from pathlib import Path

# Genre buckets (the build_eval_benchmark.py sampler stratifies on these).
GENRES = ['rock','metal','punk','pop','electronic','hiphop','jazz','country','prog','acoustic']

# Best-effort tags. Anything not listed stays empty -> the user can refine.
ARTIST_GENRE = {
    # --- metal / metalcore ---
    'August Burns Red':'metal','Attila':'metal','Ice Nine Kills':'metal',
    'Memphis May Fire':'metal','Black Label Society':'metal','Darkest Hour':'metal',
    'Crown the Empire':'metal','Dethklok':'metal','Amon Amarth':'metal',
    'Arch Enemy':'metal','In Flames':'metal','Mercyful Fate':'metal',
    'GWAR':'metal','Drowning Pool':'metal','Suicidal Tendencies':'metal',
    'Slipknot':'metal','The Amity Affliction':'metal','Woe Is Me':'metal',
    'blessthefall':'metal','The Dillinger Escape Plan':'metal',
    'Unearth':'metal','Helmet':'metal','Chimaira':'metal','Diamond Head':'metal',
    'Firewind':'metal','Saliva':'metal','Snot':'metal','Sworn':'metal',
    'John 5 Featuring Jim Root':'metal','Danzig':'metal','Motley Crue':'metal',
    'Public Enemy Featuring Zakk Wylde':'metal','Tesla':'metal','Warrant':'metal',
    'Electric Callboy':'metal','Nightwish':'metal','Get Scared':'metal',
    'Fear, And Loathing In Las Vegas':'metal','Shadow Falls':'metal',
    'Awaken':'metal','Priestess':'metal','S.U.P.R.A.H.U.M.A.N.':'metal',
    'Eternxlkz':'metal','MXZI':'metal','ORION':'metal',
    'Metallica & Ozzy Osbourne':'metal','The Damned Things':'metal',
    'Alter Bridge':'metal','Rev Theory':'metal','Michael Schenker Group':'metal',
    'Slash (With M. Shadows)':'metal','Slash (With Dave Grohl and Duff McKagan)':'metal',
    'Slash (With Iggy Pop)':'metal','Slash featuring Ian Astbury':'metal',
    'Edgar Winter':'metal','Joe Bonamassa':'metal','Jeff Beck':'metal',

    # --- punk / hardcore / emo / post-hardcore ---
    'Bayside':'punk','Box Car Racer':'punk','Four Year Strong':'punk',
    'Less Than Jake':'punk','The Story So Far':'punk','Bad Brains':'punk',
    'Strung Out':'punk','The Living End':'punk','Buzzcocks':'punk',
    'The Ramones':'punk','Senses Fail':'punk','Hawthorne Heights':'punk',
    'Madina Lake':'punk','State Champs':'punk','Foxboro Hot Tubs':'punk',
    'Story of the Year':'punk','Cute Is What We Aim For':'punk',
    'Red Jumpsuit Apparatus':'punk','I See Stars':'punk',
    'Hundred Reasons':'punk','In Her Own Words':'punk','Bowling For Soup':'punk',
    'Bowling for Soup':'punk','The Ataris':'punk','Spirit Kid':'punk',
    'Crash And The Boys':'punk','Amber Pacific':'punk','A Change of Pace':'punk',
    'The Spill Canvas':'punk','The Sleeping':'punk','Sponge':'punk',
    'Tijuana Sweetheart':'punk','Swingin\' Utters':'punk',
    'Death from Above 1979':'punk','Beatsteaks':'punk','Rx Bandits':'punk',

    # --- prog / math / djent ---
    'Tool':'prog','Polyphia':'prog','King Crimson':'prog','The Mars Volta':'prog',
    'Dredg':'prog','Protest The Hero':'prog','Steve Vai':'prog','Eric Johnson':'prog',
    'Mr. Big':'prog','Extreme':'prog',

    # --- pop ---
    'Backstreet Boys':'pop','ABBA':'pop','Demi Lovato & Joe Jonas':'pop',
    'Demi Lovato (ft Joe Jonas)':'pop','Selena Gomez & the Scene':'pop',
    'Lady Gaga (Feat. Colby O\'Donis)':'pop','Michael Jackson':'pop',
    'Hot Chelle Rae':'pop','Cobra Starship (ft. Sabi)':'pop',
    'Donny Osmond':'pop','Aly & AJ':'pop','Owl City':'pop',
    'Foster the People':'pop','High School Musical Cast':'pop',
    'Disney':'pop','Las Ketchup':'pop','LFO':'pop','Ace of Base':'pop',
    't.A.T.u.':'pop','Rick Astley':'pop','PinkPantheress':'pop',
    'Nelly Furtado':'pop','ROSÉ, Bruno Mars':'pop','Steve Perry':'pop',
    'Phil Collins':'pop','The Buggles':'pop','A Flock of Seagulls':'pop',
    'Soft Cell':'pop','Yazoo':'pop','Chumbawamba':'pop','O-Zone':'pop',
    'Vanilla Ice':'pop','The Black Eyed Peas':'pop','Sugar Ray':'pop',
    'Gym Class Heroes (ft. Adam Levine)':'pop','Janet Jackson':'pop',
    'TLC':'pop','Bell Biv DeVoe':'pop','Surf Mesa':'pop',
    'Artemas':'pop','Noah Kahan':'pop','Lil Nas X ft. Billy Ray Cyrus':'pop',
    'Sabotage & Instituto':'pop','SAINt JHN':'pop','DUKI':'pop',
    'Eslabon Armado':'pop','Hanumankind ft. Kalmi':'pop',
    'Jung Kook, Latto':'pop','Kelsy Karter':'pop','Meredith Brooks':'pop',
    'The 1975':'pop','The Aces':'pop','Wet Leg':'pop','Lana Del Rey':'pop',
    'BRADIO':'pop','Hikaru Utada':'pop','LiSA':'pop','DAY6':'pop','Puffy AmiYumi':'pop',
    'Daft Punk':'pop','Cage the Elephant':'rock',

    # --- electronic / dance ---
    'deadmau5, Rob Swire':'electronic','Technotronic':'electronic',
    'Alice Deejay':'electronic','La Roux':'electronic','The Midnight':'electronic',
    'Netsky & Hybrid Minds':'electronic','Slipstream Music':'electronic',
    'Does It Offend You, Yeah':'electronic','Housse de Racket':'electronic',

    # --- hiphop / r&b / soul ---
    'The Notorious B.I.G':'hiphop','2Pac':'hiphop','Beastie Boys ft. Kerry King':'hiphop',
    'A Tribe Called Quest':'hiphop','Wu-Tang Clan':'hiphop','Jack Harlow':'hiphop',
    'Lil Uzi Vert':'hiphop','Megan Thee Stallion':'hiphop','DMX':'hiphop',
    'Ice Spice':'hiphop','JID':'hiphop','NLE Choppa':'hiphop','Run-DMC':'hiphop',
    'Sean Paul':'hiphop','Lil Jon & the East Side Boyz':'hiphop',
    'Pop Smoke w  Lil Baby & DaBaby':'hiphop','Shop Boyz':'hiphop',
    'Terror Squad':'hiphop','Warren G':'hiphop','21 Savage':'hiphop',
    'Cypress Hill':'hiphop','Fort Minor':'hiphop','Kid Cudi':'hiphop',
    'Masked Wolf':'hiphop','The Lonely Island':'hiphop','Michael Christmas':'hiphop',
    'Otis Redding':'jazz','Marvin Gaye':'jazz','Aretha Franklin':'jazz',
    'The Isley Brothers':'jazz','Cody ChesnuTT':'jazz','Corinne Bailey Rae':'jazz',
    'Duffy':'jazz','Van Morrison':'jazz','Stray Cats':'jazz',

    # --- country / folk ---
    'Dwight Yoakam':'country','Merle Haggard':'country','Eric Church':'country',
    'Sara Evans':'country','Lucinda Williams':'country','Dolly Parton':'country',
    'John Denver':'country','Little Big Town':'country','Sworn':'metal',
    'Tony Solis':'country','Ryan Adams':'country','Brandi Carlile':'country',
    'Cracker':'country',

    # --- acoustic / singer-songwriter ---
    'Jack Johnson':'acoustic','Milky Chance':'acoustic','Seasick Steve':'acoustic',
    'Oliver Tree':'acoustic','Brian Setzer':'acoustic','Devon Gilfillian':'acoustic',
    'Benjamin Booker':'acoustic','Jack White':'acoustic',

    # --- rock (general / classic / alt) ---
    'Led Zeppelin':'rock','Tom Petty & The Heartbreakers':'rock','Black Sabbath':'rock',
    'Marcy Playground':'rock','Tonic':'rock','Highly Suspect':'rock',
    'Flogging Molly':'punk','Middle Class Rut':'rock','Foster the People':'rock',
    'Cage the Elephant':'rock','Nada Surf':'rock','Placebo':'rock',
    'Joy Division':'rock','Blue Oyster Cult':'rock','Bob Seger':'rock',
    'Neil Young':'rock','Queen & David Bowie':'rock','Queen + Paul Rodgers':'rock',
    'Ram Jam':'rock','Sponge':'rock','Tonic':'rock','The Faint':'rock',
    'The Rasmus':'rock','The Reverend Horton Heat':'rock','The Runaways':'rock',
    'The Turtles':'rock','Village People':'rock','Sponge':'rock',
    'Band Of Skulls':'rock','Blue October':'rock','Crown the Empire':'metal',
    'White Denim':'rock','The Both':'rock','The Raveonettes':'rock',
    'The Monkees (WaveGroup)':'rock','Blur (WaveGroup)':'rock',
    'Cracker':'rock','4 Non Blondes':'rock','Jesus Jones':'rock','Midnight Oil':'rock',
    'Santana (ft. Rob Thomas)':'rock','Opiate for the Masses':'rock',
    'Richard Fortus':'rock','Gov\'t Mule':'rock','Tom Petty & The Heartbreakers':'rock',
    'Red Rider':'rock','The Exies':'rock','Foxboro Hot Tubs':'punk',
    'The Duke Spirit':'rock','The Answer':'rock','The Gaslight Anthem':'rock',
    'Bloodhound Gang':'rock','Sugar Ray':'rock','Gov\'t Mule':'rock',
    'Loquillo Y Los Trogloditas':'rock','Negrita':'rock','Radio Futura':'rock',
    'Les Wampas':'punk','Les Rita Mitsouko':'rock','Orgy':'rock','Tonic':'rock',

    # --- video-game / soundtrack / chiptune (treat as rock for stratification) ---
    'Koji Kondo':'electronic','David Wise':'electronic','Kevin Sherwood':'electronic',
    'Stan Bush':'rock','The Hex Girls':'pop','Chip Skylark':'pop',
    'Nicole and Brynne Price':'pop','Jason Paige':'pop','The Toonosaurs':'pop',
    'The GAG Quartet':'rock','Mark Keali\'i Ho\'omalu, The Kamehameha Schools Children\'s Chorus':'pop',
    'Deric Battistean':'electronic','François Jalbert':'jazz','Awaken':'metal',
    'Richard Campbell':'jazz','Nancy Fullforce':'pop','Party Bois':'pop',
    'Slipstream Music':'electronic','The Rocky Horror Picture Show':'pop',
    '2. Skatune Network':'punk','3. Skatune Network':'punk','4. Skatune Network':'punk',

    # --- elvis / classics ---
    'Elvis Presley':'rock','The Four Seasons':'pop',
    'Laura Branigan':'pop','Katrina and the Waves':'pop',
    'Rod Stewart':'rock','Public Enemy Featuring Zakk Wylde':'metal',
    'Timbaland':'hiphop','Public Enemy Featuring Zakk Wylde':'metal',
    'Snot':'metal','The Toonosaurs':'pop',
}

src = Path('paper/benchmark_candidates.csv')
rows = list(csv.DictReader(src.open()))
hit = miss = 0
for r in rows:
    g = ARTIST_GENRE.get(r['artist'], '')
    if g:
        r['genre'] = g
        hit += 1
    else:
        miss += 1
print(f'tagged {hit}/{len(rows)} candidates; {miss} left untagged (user to refine)')

with src.open('w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['title','artist','genre','audio_path','midi_path','duration_s','source'])
    w.writeheader()
    w.writerows(rows)

# print per-genre count
from collections import Counter
c = Counter(r['genre'] or '(untagged)' for r in rows)
print('\nper-genre counts:')
for k, v in c.most_common():
    print(f'  {k:<14} {v}')
