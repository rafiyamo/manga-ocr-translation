# Comic Translation by AI Model

This will outline the development of an AI system that translates full pages of comics and manga from their original languages into English. This will be established through a web-based platform. This system will be developed from scratch and tailored specifically for layouts adapted for comics and/or manga. It will extract text directly from image pages, identify the reading order, and provide context aware translations. The result is expected to relay the tone and nuance of the original language.
This web application will:
1.     Accept an image, or a batch of images (PNG, JPEG, etc.) of a manga/comic page.
2.     Automatically identify and extract the text from speech bubbles, narration boxes, action lines, and sound effects.
3.     Understand the correct order in which the text should be read (e.g., right to left for Japanese).
4.     Translate the extracted text into fluent, natural English while preserving the tone, context, and meaning as much as possible.
5.     Display the translated text in a simple, readable format for users. (NOT OVERLAYED ON THE COMIC PAGE ITSELF)
 
## User Experience:
The user begins by opening the website and selecting a page from their computer, which would usually be an image file (like a .JPG or .PNG) of a manga or comic. This image is then uploaded to the system for processing.
The user might also be asked to choose the language of the comic (e.g., Japanese or Spanish) and the reading direction (e.g., right-to-left for Japanese manga).

Once the image is uploaded, the system will “read” the page, which would not just be looking at the text, but also looking at the visual layout of the comic page. 

This includes:
Finding the speech bubbles, narration boxes, and sound effects on the page.

Figuring out what order they should be read in, based on their position and direction of flow (top to bottom, left to right, or right to left).

Recognizing the text inside each bubble or box, even if it’s handwritten, distorted, or stylized.	

To do this, the system will look at shapes, patterns, and positions. It would not assume left-to-right text like a typical document scanner. After locating and reading the text in the original language, the system will start the translation process.

To stay as close to the tonality of the original language as possible, the AI translation system will try to understand the context of the page, instead of converting word by word, like typical translation models. 

For example:
If a character is shouting, or crying, or talking plainly, the translation will reflect that intensity, or calmness, and vice versa. (Bold letters, capitalized letters, italicised, etc.)
If a sound effect (like bam! or whoosh!) appears, the system will choose the best English equivalent to match the scene. (e.g. Japanese or Eastern onomatopoeia differs from its Western counterpart.)


Finally, the translated results are shown to the user in a clean, readable format. This may look like:

A numbered list of translated lines, in the correct order (matching the order you would read the comic).
\
Optional visual overlays or highlights showing which translation came from which part of the image (for reference).

A split view: original image on one side, translated text on the other. (Best choice) (combine 1. and 3. maybe)
 
## Project Pipeline:

Ø INPUT (FRONTEND)

User accesses website. Uploads multiple images (comic/manga pages) at once.

Allow batch image uploads

optional: user can specify language (e.g., Japanese, Spanish) and reading direction (e.g., right-to-left)

Show thumbnails or previews of uploaded pages

Send files to the backend for processing

Ø PREPROCESSING (BACKEND)

Convert uploaded images to standard format and size

Apply grayscale, noise reduction, contrast enhancement (for better text detection) (depends)

Use image orientation detection to ensure the page is upright

Ø TEXT DETECTION

System detects the placements and order of speech bubbles, narration boxes, sound effects on the page.

Detect text-containing regions using a model (from scratch)

Draw bounding boxes around detected areas (??)

Classify each region, label each region(?) (e.g., speech bubble, sound effect, narration)

Ø READING ORDER

Analyze relative position of each bubble or box

Sort based on reading direction (left-right or right-left, top-down)

Output an ordered list like Bubble 1 → SFX 1 → Narration → Bubble 2

Ø OCR

Raw text extracted. Train model for stylized or handwritten fonts (manga/comic specific)

Ø TRANSLATION

Translate text into fluent English trough custom built AI model.

Ø OUTPUT FORMATTING

(Possible) format as:

Page 1:

[NARRATION] “…”

[CHARACTER A: SPEECH BUBBLE] “…”

[SFX]: “…”

Ø DISPLAY RESULTS

User can download or copy the translated text. Will be provided options to download in different formats (??)
