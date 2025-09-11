from jet.logger import logger
from langchain_community.tools.google_lens import GoogleLensQueryRun
from langchain_community.utilities.google_lens import GoogleLensAPIWrapper
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger.basicConfig(filename=log_file)
logger.info(f"Logs: {log_file}")

PERSIST_DIR = f"{OUTPUT_DIR}/chroma"
os.makedirs(PERSIST_DIR, exist_ok=True)

"""
# Google Lens

This notebook goes over how to use the Google Lens Tool to fetch information on an image.

First, you need to sign up for an `SerpApi key` key at: https://serpapi.com/users/sign_up.

Then you must install `requests` with the command:

`pip install requests`

Then you will need to set the environment variable `SERPAPI_API_KEY` to your `SerpApi key`

[Alternatively you can pass the key in as a argument to the wrapper `serp_`]

## Use the Tool
"""
logger.info("# Google Lens")

# %pip install --upgrade --quiet  requests langchain-community



os.environ["SERPAPI_API_KEY"] = ""
tool = GoogleLensQueryRun(api_wrapper=GoogleLensAPIWrapper())

tool.run("https://i.imgur.com/HBrB8p0.png")

"""
Output should look like:

Subject:Danny DeVito(American actor and comedian)
Link to subject:https://www.google.com/search?q=Danny+DeVito&kgmid=/m/0q9kd&hl=en-US&gl=US

Related Images:

Title: Danny DeVito - Simple English Wikipedia, the free encyclopedia
Source(Wikipedia): https://simple.wikipedia.org/wiki/Danny_DeVito
Image: https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSm5zQhimRYYgKPVf16viNFoDSsZmGrH09dthR6cpL1DXEdzmQu

Title: File:Danny DeVito by Gage Skidmore.jpg - Wikipedia
Source(Wikipedia): https://en.m.wikipedia.org/wiki/File:Danny_DeVito_by_Gage_Skidmore.jpg
Image: https://encrypted-tbn2.gstatic.com/images?q=tbn:ANd9GcTRFc3mD4mzd3VHQFwNRK2WfFOQ38_GkzJTNbDxd1cYcN8JAc_D

Title: Danny DeVito — Wikipèdia
Source(Wikipedia): https://oc.wikipedia.org/wiki/Danny_DeVito
Image: https://encrypted-tbn3.gstatic.com/images?q=tbn:ANd9GcQNl_2mCz1tAHs_w-zkIm40bhHiuGFMOqJv9uZcxTQm9qCqC4F_

Title: US Rep. says adult animated sitcom with Danny DeVito as voice of Satan is ‘evil’
Source(wilx.com): https://www.wilx.com/2022/09/08/us-rep-adult-animated-sitcom-with-danny-devito-voice-satan-is-evil/?outputType=apps
Image: https://encrypted-tbn1.gstatic.com/images?q=tbn:ANd9GcSNpxLazAXTg09jDebFhVY0lmBgKWCKHFyqD5eCAIQrf5RI85vu

Title: Danny DeVito gets his own day in his native New Jersey
Source(WOWT): https://www.wowt.com/content/news/Danny-DeVito-gets-his-own-day-in-his-native-New-Jersey-481195051.html
Image: https://encrypted-tbn1.gstatic.com/images?q=tbn:ANd9GcTYWvHxMAm3zsMrP3vr_ML0JX2SZZkxblN_KYfxf0EI8ALuhFhf

Title: Steam Community :: Guide :: danny devito
Source(Steam Community): https://steamcommunity.com/sharedfiles/filedetails/?id=923751585
Image: https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcS-vOaIRRxi1xC7CgFUymyLzhwhnvB5evGgCNo5LlUJDiWeTlN9

Title: Danny DeVito gets his own day in his native New Jersey | News | khq.com
Source(KHQ.com): https://www.khq.com/news/danny-devito-gets-his-own-day-in-his-native-new-jersey/article_514fbbf4-7f6f-5051-b06b-0f127c82439c.html
Image: https://encrypted-tbn3.gstatic.com/images?q=tbn:ANd9GcSYN29NVlBV6L-hFKA7E2Zi32hqkItUyDUA-BtTt2fmJjwGK_Bg

Title: Mad ☆ on X: "Gavin told me I look like Danny DeVito and I can’t unsee it https://t.co/UZuUbr0QBq" / X
Source(Twitter): https://twitter.com/mfrench98/status/1062726668337496065
Image: https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTMNYrgw_ish0CEuimZ3SxU2ReJrMcEb1NVGsHNfUFy2_0v0FRM

Title: Ewan Moore on X: "I have just one casting request for the Zelda movie https://t.co/TNuU7Hpmkl" / X
Source(Twitter): https://twitter.com/EMoore_/status/1722218391644307475
Image: https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSJnljY1EpoKGpEEaptMeSdkbm1hWSb0XqLBDcWdDAmEGIWVjHw

Title: GoLocalPDX | Spotted in Portland: Danny DeVito in Pearl District
Source(GoLocalPDX): https://m.golocalpdx.com/lifestyle/spotted-in-portland-danny-devito-in-pearl-district
Image: https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSL_cpTOI7ewQCh1zDkPB7-p9b2M6d9TYX4XMKEb2j9Kwf8a4Ui

Title: Danny De Vito Meme Funny Pewdiepie Sticker | Redbubble
Source(Redbubble): https://www.redbubble.com/i/sticker/Danny-de-Vito-Meme-Funny-by-nattdrws/96554839.EJUG5
Image: https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTrIbb_rf6dK7ChbDyk5xCGTMPkNtis76m_vUYvvB_Uc3GMWqxm

Title: Danny Devito Every Day (@whydouwannakno8) / X
Source(Twitter): https://twitter.com/whydouwannakno8
Image: https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSpUx-HFqLx1FG9yphLgEU6l_SyTPaTX2FmyChtLHS3VOqXf2S5

Title: These fancasts are horrible, we all know who’d be the perfect Doomguy. : r/Doom
Source(Reddit): https://www.reddit.com/r/Doom/comments/tal459/these_fancasts_are_horrible_we_all_know_whod_be/
Image: https://encrypted-tbn2.gstatic.com/images?q=tbn:ANd9GcTPFzg9ntWpbVW3r26EMjfXVYRHO1w3c5VeeeWe1jKVmtJpSB6z

Title: Will McKinney - Hudl
Source(Hudl): https://www.hudl.com/profile/6386357
Image: https://encrypted-tbn2.gstatic.com/images?q=tbn:ANd9GcQbqpQ4wQ5qpjf0dBsgFqZW-f4FMTpePRK63BHOL_qop1D93FnK

Title: Petition · Danny DeVito to play James Bond · Change.org
Source(Change.org): https://www.change.org/p/hollywood-danny-devito-to-play-james-bond
Image: https://encrypted-tbn2.gstatic.com/images?q=tbn:ANd9GcRivkvCq6bk9OWMsWW9LAlYtf7QkYdDsJ_2skhbKstkyK9Pk07F

Title: Danny DeVito - Wikiwand
Source(Wikiwand): https://www.wikiwand.com/simple/Danny_DeVito
Image: https://encrypted-tbn3.gstatic.com/images?q=tbn:ANd9GcS4xQ_wZhK6OMttSuxsv2fjscM6la3DPNQcJt5dnWWbvQLP3CuZ

Title: Could this be the perfect actor for older Lottie? : r/Yellowjackets
Source(Reddit): https://www.reddit.com/r/Yellowjackets/comments/s5xkhp/could_this_be_the_perfect_actor_for_older_lottie/
Image: https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTaCefSusoOR5hP0pQsR3U-Ep3JVjYdr3HPjkUdut2fa1wjxHHj

Title: Pin on People who inspire me or make me giggle
Source(Pinterest): https://www.pinterest.com/pin/189080884324991923/
Image: https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcS7fbawhF3QWAZHIMgzL2W4LzW2VkTQLOB4DKUrscYnORBnuK8s

Title: Steam Curator: Official Danny Devito Fan Club
Source(Steam Powered): https://store.steampowered.com/curator/33127026-Official-Danny-Devito-Fan-Club/
Image: https://encrypted-tbn3.gstatic.com/images?q=tbn:ANd9GcTxzGbyn_8fezRf4gSNqJBq-lKXWJ8cBU-3le21vO-9fKxygBnv

Title: The Man. The Legend. : r/IASIP
Source(Reddit): https://www.reddit.com/r/IASIP/comments/h08t4n/the_man_the_legend/
Image: https://encrypted-tbn1.gstatic.com/images?q=tbn:ANd9GcSoqVN3Zd4gbZ2RdFTKy4IJnJSve_ZPmbIJOg3o5hBH5frNv3NZ

Title: Can You Match These Celebrities To Their "Simpsons" Character?
Source(BuzzFeed): https://www.buzzfeed.com/jemimaskelley/match-the-simpsons-guest-stars
Image: https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTTWkP5BwFmYmovl8ptvm1-amrhEeYPTXh19g00GKebQsuvIkkl

Title: The Adventures of Danny Devito .exe - The Adventures of Danny Devito - Wattpad
Source(Wattpad): https://www.wattpad.com/634679736-the-adventures-of-danny-devito-exe-the-adventures
Image: https://encrypted-tbn1.gstatic.com/images?q=tbn:ANd9GcTvVZ-nuX_DHP8rx6tPn3c-CoqN3O6rUKxUMzZOhiQxDIc4y2Uv

Title: Uploading Images of People That Reddit Loves Day 2 : r/memes
Source(Reddit): https://www.reddit.com/r/memes/comments/l0k5oo/uploading_images_of_people_that_reddit_loves_day_2/
Image: https://encrypted-tbn3.gstatic.com/images?q=tbn:ANd9GcRdKRC-1iyxkdHHaVEaVSkI29iMS4Ig6BBRkgX77YnsNRc8RHow

Title: Danny DeVito - Wikipedia, the free encyclopedia | Danny devito, Trending shoes, Casual shoes women
Source(Pinterest): https://www.pinterest.com/pin/170362798380086468/
Image: https://encrypted-tbn3.gstatic.com/images?q=tbn:ANd9GcTUmS49oH7BqbbCFv8Rk-blC3jFGo740PFs-4Q1R5I9p0i8GLgc

Title: Dr. Shrimp Puerto Rico on X: "y Danny de Vito como Gaetan "Mole" Moliere. https://t.co/HmblfQt2rt" / X
Source(Twitter): https://twitter.com/celispedia/status/1381361438644658183
Image: https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcThcsEyL8Vm0U2xFvZjrCoD11G6lU37PMnEVst3EfekfqC6ZC2T

Title: Why do singers shake and quiver their heads when they sing? - Quora
Source(Quora): https://www.quora.com/Why-do-singers-shake-and-quiver-their-heads-when-they-sing
Image: https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTVHZIii3O4qHE_8uIPDNf1wjCEcKho9sb40dSBiUuvA5_ffd1O

Title: New man under center for the G-Men : r/NFCEastMemeWar
Source(Reddit): https://www.reddit.com/r/NFCEastMemeWar/comments/17j8z7f/new_man_under_center_for_the_gmen/
Image: https://encrypted-tbn2.gstatic.com/images?q=tbn:ANd9GcTe2ym5Q6qlMJlcWO6ppJp3EMo3Lzl_45V-SFFh_2DZdmfaGD6k

Title: HumanSaxophone (@HumanSaxophone) / X
Source(Twitter): https://twitter.com/HumanSaxophone
Image: https://encrypted-tbn3.gstatic.com/images?q=tbn:ANd9GcQRT26qpb-YXqTUHF7VNG2FgofRQvQGGrt5PcbbhHT0uZtgZYLv

Title: 35 People Reveal What Made Them Forever Change Their Mind About Certain Celebrities | Bored Panda
Source(Bored Panda): https://www.boredpanda.com/views-changed-on-famous-person/
Image: https://encrypted-tbn3.gstatic.com/images?q=tbn:ANd9GcThO3ytsqLhlpnjYFxgz9Xu6ukfd-bR8MSSIFX8jyysZWhOpiuz

Title: How to book Danny DeVito? - Anthem Talent Agency
Source(Anthem Talent Agency): https://anthemtalentagency.com/talent/danny-devito/
Image: https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcS64Ne3byqIBuZ6RtvwCYLmQMFOneaWrF5nxfpdsNz9L7yOivu6

Title: Starring Frank Reynolds (It's Always Sunny in Philadelphia) Tag your artist friends! … | It's always sunny in philadelphia, It's always sunny, Sunny in philadelphia
Source(Pinterest): https://id.pinterest.com/pin/315181673920804792/
Image: https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRCuBvS4jdGA3_YlPX_-E4QaWnv43DXhySsJAoSy8Y_PwtHW1oC

Title: Create a Top 100 White People Tier List - TierMaker
Source(TierMaker): https://tiermaker.com/create/top-100-white-people-1243701
Image: https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTpDM6YwQpn952yLt0W8O6EIDUKRn1-4UQc0Lum2_2IFrUeZeN4

Title: P R E S S U R E | Rochelle Jordan
Source(Bandcamp): https://rochellejordan.bandcamp.com/album/p-r-e-s-s-u-r-e
Image: https://encrypted-tbn2.gstatic.com/images?q=tbn:ANd9GcTY1o_f9y5GF5lIhFl1wALTEXCU8h1HVxDQIRbxvZhd8I4u312j

Title: Danny DeVito Net Worth, Biography Age, Family, wiki, And Life Story - JAKADIYAR AREWA
Source(JAKADIYAR AREWA): https://www.jakadiyararewa.com.ng/2023/05/danny-devito-net-worth-biography-age.html
Image: https://encrypted-tbn1.gstatic.com/images?q=tbn:ANd9GcRAfAt8msNdjwKqmCP7PtgdLWxWpGfXshGiL9iF2mJ4J6MeK_oU

Title: Actors in the Most Tim Burton Movies
Source(Ranker): https://www.ranker.com/list/actors-in-the-most-tim-burton-movies/ranker-film
Image: https://encrypted-tbn3.gstatic.com/images?q=tbn:ANd9GcRh1I6T1RvdyzauITQ4CcZheqCorQtfZZt9w_-b7-l9gjD6E8dy

Title: File:Danny DeVito 2011.jpg - Wikimedia Commons
Source(Wikimedia): https://commons.wikimedia.org/wiki/File:Danny_DeVito_2011.jpg
Image: https://encrypted-tbn3.gstatic.com/images?q=tbn:ANd9GcR81S9hnwqjxwtyAGx5HmDLGealisuAt8m-f2baNLgJroxheFi0

Title: Warlock | A D&D Audio Drama⭐ on Twitter: "Easy, Gandalf! #lotr https://t.co/XOwnQD0uVd" / X
Source(Twitter): https://twitter.com/Warlockdnd/status/1462582649571139586
Image: https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQUZ-666ydtuN02MSkM32w-61j9cpIIrXI8bWsKAJRzG3irR8Yg

Title: Pin by Sarah Richardson on nice photos of danny devito | Danny devito, Celebrity caricatures, Cute celebrities
Source(Pinterest): https://www.pinterest.com/pin/600526931536339674/
Image: https://encrypted-tbn2.gstatic.com/images?q=tbn:ANd9GcSoMQ0XnsrNUqpXNgKeAyjXX4PgNlCdJksiAv23Y0h4w_Kn2SUO

Title: Is Jennifer Lawrence Jewish? - Wondering: Is Danny DeVito Jewish?
Source(Blogger): http://jenniferlawrencejewishwondering.blogspot.com/2012/02/is-danny-devito-jewish.html
Image: https://encrypted-tbn2.gstatic.com/images?q=tbn:ANd9GcTQjUbutXtyO4Vv9g3cRjc8IF5h8IKO-3JvpNJDm-WR40fwtUTz

Title: Randorfizz Stories - Wattpad
Source(Wattpad): https://www.wattpad.com/stories/randorfizz
Image: https://encrypted-tbn1.gstatic.com/images?q=tbn:ANd9GcSuaG_WJmQqIXTBqHAQsim0LiOQrmtLAT-DSrJ0wsWLGnfrOgiC

Title: What is the name of the agreement that laid the foundation for a limited monarchy in England? - brainly.com
Source(Brainly): https://brainly.com/question/7194019
Image: https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTsasmT8IK-Pasa10LGayrjgmxerp80HuFThhfTCut3m4hSPM4F

Title: Find an Actor to Play Danny DeVito in The Beatles Yellow Submarine [ Remake ] on myCast
Source(myCast.io): https://www.mycast.io/stories/the-beatles-yellow-submarine-remake/roles/danny-devito-1/6531784/cast
Image: https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRu8vm6Po98ZACAXzithjj6yKDxhQtgKPDC6rSKLMcFfhv8FreR

Title: Journey's End Vanity Contest Submission Thread | Page 301 | Terraria Community Forums
Source(Terraria): https://forums.terraria.org/index.php?threads/journeys-end-vanity-contest-submission-thread.86457/page-301
Image: https://encrypted-tbn2.gstatic.com/images?q=tbn:ANd9GcTjsnksAzRqRhoH1SSxHTk7uBjhzLjHl-EZyKN8gI1kzTNO3irh

Title: Better Characters… : r/TheMandalorianTV
Source(Reddit): https://www.reddit.com/r/TheMandalorianTV/comments/11wi6z6/better_characters/
Image: https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcR6DeMvwDob6F149S84_jRNw4kkVfVFQiFi1tnDVghTMJv1ghHw

Title: Top 5 Bald Men Style Tips- a guide to how to rock the bald look
Source(asharpdressedman.com): https://asharpdressedman.com/top-5-bald-men-style-tips/
Image: https://encrypted-tbn2.gstatic.com/images?q=tbn:ANd9GcRr1fIuLo78h_-LgRGk6R5dyt3jk9eloSNuqWKA-Xb_4aTuB0yh

Title: Danny DeVito Facts for Kids
Source(Kiddle): https://kids.kiddle.co/Danny_DeVito
Image: https://encrypted-tbn1.gstatic.com/images?q=tbn:ANd9GcRa0rikFTYgSgOyt3XuVtFg4qvPY5xzOABgXi8Kx0y9wdvHTHJa

Title: Total Drama Fan-casting - Duncan : r/Totaldrama
Source(Reddit): https://www.reddit.com/r/Totaldrama/comments/111c9wi/total_drama_fancasting_duncan/
Image: https://encrypted-tbn3.gstatic.com/images?q=tbn:ANd9GcSzRzJmkh0NJqG1eHky0jCyzlje8ZVF8GMVIS0F6NjzTOTAWZas

Title: Danny DeVito - Alchetron, The Free Social Encyclopedia
Source(Alchetron.com): https://alchetron.com/Danny-DeVito
Image: https://encrypted-tbn3.gstatic.com/images?q=tbn:ANd9GcTBL-5gHoQCIQ9nftiTBrHtKb0hQftD5FkZaBexyKJVfFBa8gEI

Title: Which of these Acts forced American colonists to allow British troops to stay in their homes? the - brainly.com
Source(Brainly): https://brainly.com/question/19184876
Image: https://encrypted-tbn2.gstatic.com/images?q=tbn:ANd9GcR5efzmJVyU63yHNOrtHtr7HqY2fA7R3i_h4GqmGmQAjnRwULNo

Title: Nathan Heald - Bettendorf, Iowa | about.me
Source(About.me): https://about.me/nathanheald2020
Image: https://encrypted-tbn3.gstatic.com/images?q=tbn:ANd9GcT9oNTZAOVsfDYlvne3MS9Uk6utafVrOcMwBxfXuI1qLLpd4Yvk

Title: Dannydevito Stories - Wattpad
Source(Wattpad): https://mobile.wattpad.com/stories/dannydevito/new
Image: https://encrypted-tbn1.gstatic.com/images?q=tbn:ANd9GcT15bfDZnlFZZNWytOFpDYe3JgKr8H0Nccm7Dt_2KfsqHDK0KnH

Title: Drunk Celebrities | Crazy Things Famous People Have Done While Drunk
Source(Ranker): https://www.ranker.com/list/things-celebrities-have-done-drunk/celebrity-lists
Image: https://encrypted-tbn3.gstatic.com/images?q=tbn:ANd9GcTfX2sB59QDDJMuBcSXR9gvpkBjCDiHacCLRq9SYSBdj-apAecM

Title: Actress Jessica Walter and Aisha Tyler of the television show... News Photo - Getty Images
Source(Getty Images): https://www.gettyimages.ca/detail/103221172
Image: https://encrypted-tbn2.gstatic.com/images?q=tbn:ANd9GcTwB7RxA0jvAOWhas8KCl3im7viaTuha3jJcd2O-cR2oUMh9mPx

Title: Jones BBQ and Foot Massage, W 119th St, Chicago, IL, Barbecue restaurant - MapQuest
Source(MapQuest): https://www.mapquest.com/us/illinois/jones-bbq-and-foot-massage-427925192
Image: https://encrypted-tbn3.gstatic.com/images?q=tbn:ANd9GcSN7Ril--htuGdToqlbVnozBNw07F4iRziioDb6l4iB-XR2Ut5z

Title: Danny Devito | Made up Characters Wiki | Fandom
Source(Fandom): https://muc.fandom.com/wiki/Danny_Devito
Image: https://encrypted-tbn3.gstatic.com/images?q=tbn:ANd9GcTOP6c2mD5E_r5Ni_kBVWnWUuud3rKsq7dDNxK2pyEW1NgCrUoR

Title: Not even sorry : r/2westerneurope4u
Source(Reddit): https://www.reddit.com/r/2westerneurope4u/comments/1510k3o/not_even_sorry/
Image: https://encrypted-tbn2.gstatic.com/images?q=tbn:ANd9GcRvjrraaXuyKTBNM9jcElIizdl7zV7TjunI3BmPPyEQDWd5fQC8

Title: Eduardo García-Molina on X: "Quintus, fetch the oil container shaped like a satyr that resembles Danny Devito. https://t.co/ykq7DjYNsw" / X
Source(Twitter): https://twitter.com/eduardo_garcmol/status/1529073971924197379
Image: https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQ8exTZLs7tS5A5hRHu1mzfcxF_HCFmFJjI8i1_s6CNrv-6880C

Title: Over 10k People Have Signed A Petition To Make Danny DeVito The New Wolverine | Bored Panda
Source(Bored Panda): https://www.boredpanda.com/petition-danny-devito-wolverine-mcu/
Image: https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQkZH-q5fAaWJxLFqcdF0UF9330mew-ZcaP5kHV777SsBOvp5C0

Title: 25 Celebrities Who Had Strange Jobs Before Becoming Famous
Source(List25): https://list25.com/25-celebrities-who-had-strange-jobs-before-becoming-famous/
Image: https://encrypted-tbn1.gstatic.com/images?q=tbn:ANd9GcT_vmlaNBdScdL2Izbw1ZxZ3CdtR3-GHB1v1CHGjSAoF0TZbKHu

Title: Devito Stories - Wattpad
Source(Wattpad): https://www.wattpad.com/stories/devito
Image: https://encrypted-tbn3.gstatic.com/images?q=tbn:ANd9GcSi5b1ySjaeTJ03fpTaLEywhm4tIK3V09PNbSUxPzJXbYJAzI4U

Reverse Image Search Link: https://www.google.com/search?tbs=sbi:AMhZZiv9acCYDkXLdR2-t3B1NkMkwOSRU-HfCIRFpYNWIVV2HdvcQJXAXmrouFitURVBkGChb8nYqHanJy4XqFL0fwt_195TZ2y0pnWZpmvecdawnkL2pwu-4F7H09e9b6SVe3Gb9fGljXuTAL8jUXOEv078EfxLyQA
"""
logger.info("Output should look like:")

logger.info("\n\n[DONE]", bright=True)