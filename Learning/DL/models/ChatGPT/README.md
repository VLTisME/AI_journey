##
## Stages
1. Pretraining
- Download -> preprocess the internet. HuggingFace curated ton of texts, documents to create FineWeb dataset. They are high quality datasets. Pipeline is like this:
[![image.png](https://i.postimg.cc/j2t3mdrp/image.png)](https://postimg.cc/njS1Qxr2)
- FineWeb used CommonCrawl (CC) as a starting point (it crawls all pages), the CC is kinda raw. Then Fineweb document it, preprocess it:
1.1. URL filtering
- block adult contents, malware websites, racist websites,...
1.2 text extraction
- since we crawl raw HTML, or,.... -> we extract the text of the web page.
1.3 Language filtering
- Use fastText language classifier to keep only English text with a score >= 0.65 -> then the model after training on Fineweb will be very good at english but not the other languages.
- 
