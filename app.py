import os
import subprocess
import time
import requests
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv
from threading import Thread
from concurrent.futures import ThreadPoolExecutor

from flask import Flask, render_template, request, redirect, url_for, jsonify, session
from werkzeug.utils import secure_filename

import imageio_ffmpeg as ffmpeg

# === Load API Keys ===
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")
search_engine_id = os.getenv("SEARCH_ENGINE_ID")

client = OpenAI(api_key=openai_api_key)

# === Flask App Setup ===
app = Flask(__name__)
app.secret_key = os.urandom(24)

VIDEO_FOLDER = os.path.join("static", "videos")
AUDIO_FOLDER = os.path.join("static", "audios")
os.makedirs(VIDEO_FOLDER, exist_ok=True)
os.makedirs(AUDIO_FOLDER, exist_ok=True)

# === Global progress & results trackers ===
processing_progress = {}
processing_results = {}
processing_cancelled = {}  # Track cancelled processes

# === Caching for performance ===
search_cache = {}  # Cache Google search results to avoid duplicate API calls

# === Processing Functions ===
def convert_video_to_transcript(video_path, filename):
    processing_progress[filename] = 10
    print("Converting video to audio...")

    base_name = os.path.splitext(os.path.basename(video_path))[0]
    audio_path = os.path.join(AUDIO_FOLDER, base_name + "_audio.mp3")

    # Get the FFmpeg executable bundled with imageio-ffmpeg
    ffmpeg_path = ffmpeg.get_ffmpeg_exe()

    # FFmpeg command
    ffmpeg_cmd = [
        ffmpeg_path,
        "-i", video_path,
        "-vn",               # no video
        "-acodec", "libmp3lame",
        "-ab", "128k",       # bitrate
        "-ar", "16000",      # sample rate
        "-ac", "1",          # mono audio
        "-y", audio_path
    ]

    try:
        subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
        print(f"Audio extracted: {audio_path}")
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Audio extraction failed.\n{e}")
        return "", ""

    if not os.path.exists(audio_path):
        print("ERROR: Audio file not found.")
        return "", ""

    processing_progress[filename] = 20
    print("Transcribing audio...")

    # Whisper API call
    with open(audio_path, "rb") as audio_file:
        try:
            transcript_response = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="text"
            )
            transcript_text = transcript_response.text if hasattr(transcript_response, 'text') else str(transcript_response)
        except Exception as e:
            print(f"Whisper API failed: {e}")
            return "", ""

    processing_progress[filename] = 30
    return transcript_text, audio_path


def summarize_transcript(transcript, filename, language='english'):
    processing_progress[filename] = 40
    prompt = (
        "You are a TikTok health misinformation fact-checker.\n "
        "Your job is to summarize the TikTok transcript.\n"
        "Always starts with 'The video discusses' or any similar phrase.\n"
        "Summarize the health-related content contained within.\n"
        "If the video is not health-related, state that clearly and do not do any further processing. do not summarize and tell the user the reason is because it is not health-related.\n"
        "State the speaker's personal opinion or viewpoint if available.\n"
        "Ignore jokes, stories, or unrelated intros/outros.\n"
        "Use neutral, formal, and easy vocabulary.\n"
        "If the speaker references a source (e.g., says 'according to'), mention that.\n"
        "Use **bold** for critical medical terms, conditions, or treatments mentioned.\n"
        "Use ==highlight== for crucial findings, conclusions, or warnings that viewers should pay attention to.\n"
        "Only highlight the most important 2-3 points that summarize the core message.\n"
        "If the speaker uses any specific terminology or jargon, include that exactly as stated.\n"
        "If the summary contains a list of items:\n"
        "1. Present them as a proper numbered list\n"
        "2. Each item should be on a new line\n"
        "3. Preserve proper formatting and indentation\n"
        "4. Add a blank line before and after the list\n"
    )
    
    # Add language instruction if Malay is selected
    if language == 'malay':
        prompt += "\nIMPORTANT: Provide your entire response in Bahasa Melayu (Malay language).\n"
        "Start with 'Video ini membincangkan' instead of 'The video discusses'."

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": transcript},
        ],
    )
    processing_progress[filename] = 50
    return response.choices[0].message.content


def extract_claims(transcript, filename, language='english'):
    processing_progress[filename] = 55
    prompt = (
        "You are a TikTok health misinformation fact-checker.\n"
        "Extract all public-health-related claims made in this transcript.\n"
        "If the video is not health-related, state that clearly and do not do any further processing. do not extract claims and tell the user the reason is because it is not health-related.\n"
        "Include health, wellness, fitness, nutrition, etc.\n"
        "Write each claim clearly and make it easy to understand, keeping wording close to original.\n"
        "Exclude questions, but include opinions as claims.\n"
        "Include vague or unverifiable claims.\n"
        "Combine repeated claims extracted in the transcript into one claim.\n"
        "Return as a list."
    )
    
    # Add language instruction if Malay is selected
    if language == 'malay':
        prompt += "\nIMPORTANT: Provide your entire response in Bahasa Melayu (Malay language)."

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": transcript},
        ],
    )

    raw_claims = response.choices[0].message.content.strip().split("\n")
    claims = [c.lstrip("0123456789.-) ").strip() for c in raw_claims if c.strip()]
    processing_progress[filename] = 60
    return claims


def search_google(query):
    #check cache first to avoid duplicate API calls
    cache_key = query.lower().strip()
    if cache_key in search_cache:
        print(f"Using cached results for: {query}")
        return search_cache[cache_key]
    
    #optimized Google search with fewer results and timeout
    url = f"https://www.googleapis.com/customsearch/v1?q={query}&key={google_api_key}&cx={search_engine_id}&num=5"
    
    try:
        response = requests.get(url, timeout=10)  #add timeout for speed
        if response.status_code == 200:
            results = response.json().get('items', [])
            print(f"Found {len(results)} search results for: {query}")
            
            #cache the results
            search_cache[cache_key] = results
            
            #prevent cache from growing too large
            if len(search_cache) > 100:
                # Remove oldest 20 entries
                oldest_keys = list(search_cache.keys())[:20]
                for key in oldest_keys:
                    del search_cache[key]
            
            return results
        else:
            print(f"Google API Error: {response.status_code}")
            search_cache[cache_key] = []
            return []
    except requests.RequestException as e:
        print(f"Search request failed: {e}")
        search_cache[cache_key] = []
        return []


def gather_facts_from_sources(claim):
    results = search_google(claim)
    facts = ""

    sources_info = []
    for i, item in enumerate(results[:3], 1):
        title = item.get("title", "").split(" - ")[0].strip()  # Get main source name
        link = item.get("link", "")
        snippet = item.get("snippet", "")
        facts += f"[{title}] {snippet}\n\n"
        sources_info.append({
            "id": i,
            "name": title,
            "title": item.get("title", ""),
            "link": link,
            "snippet": snippet
        })

    return facts.strip(), sources_info


def fact_check_claim(claim, facts, language='english'):
    print(f"Verifying claim — '{claim}'")

    prompt = (
        "You are a strict health misinformation fact-checker.\n"
        "Your job is to verify a health-related claim using reliable, trustworthy sources.\n"
        "If the video is not health-related, state that clearly and do not do any further processing. do not fact check and tell the user the reason is because it is not health-related.\n"
        "Use ONLY the facts provided from a reliable search engine. Do not guess.\n\n"
        "Instructions:\n"
        "- Start your response with exactly one word: TRUE, FALSE, or UNCERTAIN (in uppercase).\n"
        "- Follow immediately with a detailed explanation on a new line.\n"
        "- Do NOT repeat the verdict word in the explanation.\n"
        "- The explanation should use neutral, formal, and simple vocabulary.\n"
        "- Use **text** for:\n"
        "  * Medical terms and conditions\n"
        "  * Scientific terminology\n"
        "  * Treatment names\n"
        "  * Technical procedures\n"
        "- Use ==highlight== ONLY for:\n"
        "  * Key findings that directly support/refute the claim\n"
        "  * Statistical evidence and research results\n"
        "  * Important warnings or precautions\n"
        "  * Direct contradictions to the claim\n"
        "- If facts are unclear, conflicting, or missing, choose UNCERTAIN and explain why.\n"
        "- If the claim is supported by multiple sources, explicitly state this.\n"
        "- For misleading/illogical claims, provide clear evidence-based refutation.\n"
        "- Ignore low-quality sources (e.g. personal blogs, unreliable forums).\n\n"
        "RESPONSE/OUTPUT FORMAT (follow exactly):\n"
        "TRUE/FALSE/UNCERTAIN.\n"
        "[Explanation with citations]\n"
        "\n"
        "Examples:\n"
        "TRUE.\n"
        "==Multiple clinical studies== have confirmed that **vitamin D supplementation** can improve bone density. Research from the ==Mayo Clinic shows a 25% reduction in fracture risk== among elderly patients taking **800 IU daily**.\n"
        "\n"
        "FALSE.\n"
        "==A comprehensive review of 15 clinical trials== found no evidence supporting this claim. **Conventional treatments** remain the most effective approach, with ==success rates of 85%== according to recent medical data. Users should ==exercise caution== as this method may pose health risks.\n"
        "\n"
        "UNCERTAIN.\n"
        "While initial **clinical trials** show ==promising results== for this treatment, ==larger studies are needed== to confirm its effectiveness. The current evidence is limited and **research methodology** needs improvement."
    )
    
    # Add language instruction if Malay is selected
    if language == 'malay':
        prompt += "\nIMPORTANT: Provide your entire response in Bahasa Melayu (Malay language).\n"
        prompt += "Use 'BENAR' instead of 'TRUE', 'PALSU' instead of 'FALSE', and 'TIDAK PASTI' instead of 'UNCERTAIN'."

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"Claim: {claim}\n\nFacts:\n{facts}"},
        ],
    )

    return response.choices[0].message.content.strip()


def log_message(filename, message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{filename}] {message}")


def process_video_fact_check(video_path, filename, language='english'):
    log_message(filename, "Processing started.")
    processing_progress[filename] = 5
    processing_cancelled[filename] = False  # Initialize cancellation flag
    start_time = time.time()

    # === Transcription ===
    if processing_cancelled.get(filename, False):
        return {"error": "Processing cancelled"}
    
    log_message(filename, "Starting transcription...")
    transcript, audio_path = convert_video_to_transcript(video_path, filename)
    if not transcript:
        processing_progress[filename] = -1
        log_message(filename, "Transcription failed.")
        return {"error": "Transcription failed."}
    log_message(filename, "Transcription completed.")
    print("\nTRANSCRIPT:\n", transcript, "\n")

    # === Summary ===
    if processing_cancelled.get(filename, False):
        return {"error": "Processing cancelled"}
        
    log_message(filename, "Summarizing transcript...")
    summary = summarize_transcript(transcript, filename, language)
    log_message(filename, "Summary completed.")
    print("SUMMARY:\n", summary, "\n")

    # === Claim Extraction ===
    if processing_cancelled.get(filename, False):
        return {"error": "Processing cancelled"}
        
    log_message(filename, "Extracting claims...")
    claims = extract_claims(transcript, filename, language)
    log_message(filename, f"{len(claims)} claim(s) extracted.")
    print("CLAIMS:")
    for i, claim in enumerate(claims, 1):
        print(f"  Claim {i}: {claim}")
    print()

    # === Fact-checking with optimized parallel processing ===
    verdicts = []
    sources = []
    num_claims = len(claims)

    if num_claims == 0:
        processing_progress[filename] = 95
    elif num_claims == 1:
        # Single claim - process sequentially
        claim = claims[0]
        if claim.strip():
            log_message(filename, f"Fact-checking claim: {claim}")
            processing_progress[filename] = 70
            
            facts, sources_info = gather_facts_from_sources(claim)
            verdict = fact_check_claim(claim, facts, language)
            verdicts.append({
                'text': verdict,
                'sources': sources_info
            })
            sources.extend([s['link'] for s in sources_info])
    else:
        # Multiple claims - use controlled parallel processing
        def process_single_claim(claim_data):
            try:
                i, claim = claim_data
                if processing_cancelled.get(filename, False):
                    return None
                    
                if not claim.strip():
                    return None

                log_message(filename, f"Fact-checking claim {i}/{num_claims}: {claim}")
                
                # Thread-safe progress update
                progress = 60 + int(35 * ((i - 1) / max(1, num_claims - 1)))
                processing_progress[filename] = progress

                facts, sources_info = gather_facts_from_sources(claim)
                verdict = fact_check_claim(claim, facts, language)
                
                return {
                    'verdict': {
                        'text': verdict,
                        'sources': sources_info
                    },
                    'links': [s['link'] for s in sources_info],
                    'index': i
                }
            except Exception as e:
                log_message(filename, f"Error processing claim {i}: {str(e)}")
                return None

        # Use parallel processing with limited workers to avoid rate limits
        max_workers = min(3, num_claims)  # Max 3 concurrent requests
        
        try:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                claim_items = [(i+1, claim) for i, claim in enumerate(claims)]
                results = list(executor.map(process_single_claim, claim_items))
        except Exception as e:
            log_message(filename, f"Parallel processing failed, falling back to sequential: {str(e)}")
            # Fallback to sequential processing
            results = []
            for i, claim in enumerate(claims, 1):
                if processing_cancelled.get(filename, False):
                    break
                result = process_single_claim((i, claim))
                results.append(result)
        
        # Collect results in order
        for result in results:
            if result is not None:
                verdicts.append(result['verdict'])
                sources.extend(result['links'])  # Still keep track of all sources

    end_time = time.time()
    total_seconds = round(end_time - start_time)
    minutes, seconds = divmod(total_seconds, 60)
    processing_time = f"{minutes}m {seconds}s"

    processing_progress[filename] = 100
    log_message(filename, f"Processing completed in {processing_time}.")

    return {
        "video_url": "/" + video_path.replace("\\", "/"),
        "audio_url": "/" + audio_path.replace("\\", "/"),
        "transcript": transcript,
        "summary": summary,
        "claims": claims,
        "verdicts": verdicts,
        "sources": sources,
        "processing_time": processing_time,
    }


@app.route("/", methods=["GET"])
def index():
    return render_template("home_upload_page.html")


@app.route("/upload", methods=["POST"])
def upload():
    if 'video' not in request.files:
        return jsonify({"status": "error", "message": "No video uploaded"}), 400

    video = request.files['video']
    language = request.form.get('language', 'english')  # Get language preference
    filename = secure_filename(video.filename)
    video_path = os.path.join(VIDEO_FOLDER, filename)
    video.save(video_path)

    processing_progress[filename] = 0

    def background_processing():
        result = process_video_fact_check(video_path, filename, language)
        processing_results[filename] = result

    Thread(target=background_processing).start()

    return jsonify({"status": "uploaded", "filename": filename})


@app.route("/processing_status", methods=["GET"])
def processing_status():
    filename = request.args.get("filename")
    if not filename:
        return jsonify({"status": "error", "message": "Filename missing"}), 400

    progress = processing_progress.get(filename)
    if progress is None:
        return jsonify({"status": "error", "message": "No such file processing"}), 404

    if progress == 100:
        # Return processing time if available
        result = processing_results.get(filename)
        processing_time = None
        if result and "processing_time" in result:
            processing_time = result["processing_time"]
        return jsonify({"status": "done", "progress": 100, "processing_time": processing_time})

    if progress == -1:
        return jsonify({"status": "error", "message": "Processing failed."})
    
    if progress == -2:
        return jsonify({"status": "cancelled", "message": "Processing was cancelled."})

    # Add estimated time remaining based on progress
    response = {"status": "processing", "progress": progress}
    
    # Optional: Add stage information for better UX
    if progress <= 30:
        response["stage"] = "transcription"
    elif progress <= 50:
        response["stage"] = "summarization"
    elif progress <= 60:
        response["stage"] = "claim_extraction"
    else:
        response["stage"] = "fact_checking"
    
    return jsonify(response)


@app.route("/cancel_processing", methods=["POST"])
def cancel_processing():
    filename = request.json.get("filename")
    if not filename:
        return jsonify({"status": "error", "message": "Filename missing"}), 400
    
    # Set cancellation flag
    processing_cancelled[filename] = True
    
    # Clean up progress tracking
    if filename in processing_progress:
        processing_progress[filename] = -2  # Special code for cancelled
    
    log_message(filename, "Processing cancelled by user.")
    return jsonify({"status": "cancelled", "message": "Processing cancelled"})


@app.route("/output", methods=["GET"])
def output():
    # Get filename from query parameter for the new system
    filename = request.args.get("filename")
    if filename and filename in processing_results:
        result = processing_results[filename]

    return render_template(
        "result_page.html",
        video_url=result["video_url"],
        audio_url=result["audio_url"],
        transcript=result["transcript"],
        summary=result["summary"],
        claims=result["claims"],
        verdicts=result["verdicts"],
        sources=result["sources"],
        processing_time=result["processing_time"],
        zipped_claims_verdicts=zip(result["claims"], result["verdicts"]),
    )


if __name__ == "__main__":
    app.run(debug=True)


