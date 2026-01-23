# JetScripts/audio/speech/run_label_speakers.py
import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Union

from rich.console import Console
from rich.table import Table

from jet.audio.speech.pyannote.segment_speaker_labeler import SegmentResult, SegmentSpeakerLabeler
from jet.audio.utils import resolve_audio_paths
from jet.file.utils import save_file
from jet.logger import logger


BASE_OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(BASE_OUTPUT_DIR, ignore_errors=True)
BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run_agglomerative_clustering(
    segment_paths: List[str],
) -> Tuple[List[SegmentResult], bool, float]:
    """Run blind agglomerative clustering (estimates number of speakers automatically)."""
    labeler = SegmentSpeakerLabeler()

    cluster_results = labeler.cluster_segments(segment_paths)
    similarity = labeler.similarity(segment_paths[0], segment_paths[1])
    same_speaker = labeler.is_same_speaker(segment_paths[0], segment_paths[1])

    return cluster_results, same_speaker, similarity


def run_reference_assignment(
    segment_paths: List[str],
    hf_token: str | None,
    reference_paths_by_speaker: Dict[Union[int, str], List[str]],
    assignment_threshold: float = 0.68,
) -> List[SegmentResult]:
    """Run reference-guided speaker assignment (supervised / anchored mode)."""
    output_dir = BASE_OUTPUT_DIR / "reference_assignment"
    output_dir.mkdir(parents=True, exist_ok=True)

    labeler = SegmentSpeakerLabeler(
        reference_paths_by_speaker=reference_paths_by_speaker,
        assignment_threshold=assignment_threshold,
        assignment_strategy="centroid",
    )

    results = labeler.cluster_segments(segment_paths)

    save_file(results, output_dir / f"results_threshold_{assignment_threshold}.json")

    return results


def get_reference_mapping(
    refs_son: List[str],
    refs_mom: List[str]
) -> Dict[int, List[str]]:
    """Define speaker reference groups → label mapping (central place to change semantics)."""
    return {
        0: refs_son,   # sons   → label 0
        1: refs_mom    # moms   → label 1
    }


def print_summary(results: List[SegmentResult], title: str) -> None:
    """Display clean rich table summary of speaker assignment results."""
    console = Console()
    table = Table(title=title)
    table.add_column("Parent Directory")
    table.add_column("Speaker Label")
    table.add_column("Centroid Cos Sim")
    table.add_column("N-neighbor Cos Sim")

    for res in results:
        table.add_row(
            res["parent_dir"],
            str(res["speaker_label"]),
            f"{res['centroid_cosine_similarity']:.3f}",
            f"{res['nearest_neighbor_cosine_similarity']:.3f}",
        )

    console.print(table)


def main() -> None:
    HF_TOKEN = os.getenv("HF_TOKEN")  # Make sure HF_TOKEN is set in your environment

    # # Reference segments (clean clips for each speaker group)
    # refs_son = [
    #     "/Users/jethroestrada/Desktop/External_Projects/Jet_Windows_Workspace/servers/live_subtitles/generated/live_subtitles_client_with_overlay/segments/segment_0001/sound.wav",
    #     "/Users/jethroestrada/Desktop/External_Projects/Jet_Windows_Workspace/servers/live_subtitles/generated/live_subtitles_client_with_overlay/segments/segment_0002/sound.wav",
    #     "/Users/jethroestrada/Desktop/External_Projects/Jet_Windows_Workspace/servers/live_subtitles/generated/live_subtitles_client_with_overlay/segments/segment_0003/sound.wav",
    #     "/Users/jethroestrada/Desktop/External_Projects/Jet_Windows_Workspace/servers/live_subtitles/generated/live_subtitles_client_with_overlay/segments/segment_0004/sound.wav",
    #     "/Users/jethroestrada/Desktop/External_Projects/Jet_Windows_Workspace/servers/live_subtitles/generated/live_subtitles_client_with_overlay/segments/segment_0005/sound.wav",
    #     "/Users/jethroestrada/Desktop/External_Projects/Jet_Windows_Workspace/servers/live_subtitles/generated/live_subtitles_client_with_overlay/segments/segment_0006/sound.wav",
    # ]
    # refs_mom = [
    #     "/Users/jethroestrada/Desktop/External_Projects/Jet_Windows_Workspace/servers/live_subtitles/generated/live_subtitles_client_with_overlay/segments/segment_0007/sound.wav",
    #     "/Users/jethroestrada/Desktop/External_Projects/Jet_Windows_Workspace/servers/live_subtitles/generated/live_subtitles_client_with_overlay/segments/segment_0008/sound.wav",
    #     "/Users/jethroestrada/Desktop/External_Projects/Jet_Windows_Workspace/servers/live_subtitles/generated/live_subtitles_client_with_overlay/segments/segment_0009/sound.wav",
    # ]

    segments_dir = "/Users/jethroestrada/Desktop/External_Projects/Jet_Windows_Workspace/servers/live_subtitles/generated/live_subtitles_client_with_overlay_2_speakers/segments"
    all_segments = resolve_audio_paths(segments_dir, recursive=True)
    same_segments = all_segments[2:4]
    different_segments = all_segments[0:2]

    # # -------------------------------------------------------------------------
    # # Blind (unsupervised) agglomerative clustering runs - increasing window
    # # -------------------------------------------------------------------------
    # print("\n" + "="*60)
    # print("BLIND AGGLOMERATIVE CLUSTERING (unsupervised)")
    # print("="*60)

    # # Filter windows with > 1 segments
    # increasing_window_segments = [
    #     window for window in increasing_window(all_segments, step_size=1)
    #     if len(window) > 1
    # ]

    # for window_num, refs_window in enumerate(increasing_window_segments, start=1):
    #     logger.info(f"\nRunning window {window_num} ({len(refs_window)} segments)...")
    #     agg_results = run_agglomerative_clustering(refs_window, HF_TOKEN, distance_threshold=0.7)
    #     print_summary(agg_results, f"Window {window_num} ({len(refs_window)} segments)")
    #     print()

    # -------------------------------------------------------------------------
    # Blind (unsupervised) agglomerative clustering runs
    # -------------------------------------------------------------------------
    print("\n" + "="*60)
    print("BLIND AGGLOMERATIVE CLUSTERING (unsupervised)")
    print("="*60)

    logger.info("\nRunning on same segments...")
    agg_results, same_speaker, similarity = run_agglomerative_clustering(same_segments)
    print_summary(agg_results, "Same segments")
    save_file(agg_results, BASE_OUTPUT_DIR / "cluster_same.json")
    save_file({
        "same_speaker": same_speaker,
        "similarity": similarity,
    }, BASE_OUTPUT_DIR / "meta_same_first_2.json")

    logger.info("\nRunning on different segments...")
    agg_results, same_speaker, similarity = run_agglomerative_clustering(different_segments)
    print_summary(agg_results, "Different segments")
    save_file(agg_results, BASE_OUTPUT_DIR / "cluster_diff.json")
    save_file({
        "same_speaker": same_speaker,
        "similarity": similarity,
    }, BASE_OUTPUT_DIR / "meta_diff_first_2.json")

    # print("\nRunning 1 - 2 sons...")
    # agg_results = run_agglomerative_clustering([refs_son[0], refs_son[1]], HF_TOKEN, distance_threshold=0.7)
    # print_summary(agg_results, "1 – Agglomerative (blind) - 2 sons")

    # print("\nRunning 2 - 2 moms...")
    # agg_results = run_agglomerative_clustering([refs_mom[0], refs_mom[1]], HF_TOKEN, distance_threshold=0.7)
    # print_summary(agg_results, "2 – Agglomerative (blind) - 2 moms")

    # print("\nRunning 3 - all sons...")
    # agg_results = run_agglomerative_clustering(refs_son, HF_TOKEN, distance_threshold=0.7)
    # print_summary(agg_results, "3 – Agglomerative (blind) - all sons")

    # print("\nRunning 4 - all moms...")
    # agg_results = run_agglomerative_clustering(refs_mom, HF_TOKEN, distance_threshold=0.7)
    # print_summary(agg_results, "4 – Agglomerative (blind) - all moms")

    # print("\nRunning 5 - all sons + 1 mom...")
    # agg_results = run_agglomerative_clustering([*refs_son, refs_mom[0]], HF_TOKEN, distance_threshold=0.7)
    # print_summary(agg_results, "5 – Agglomerative (blind) - all sons + 1 mom")

    # print("\nRunning 6 - all moms + 1 son...")
    # agg_results = run_agglomerative_clustering([*refs_mom, refs_son[0]], HF_TOKEN, distance_threshold=0.7)
    # print_summary(agg_results, "6 – Agglomerative (blind) - all moms + 1 son")

    # print("\nRunning 7 - all sons + all moms...")
    # agg_results = run_agglomerative_clustering([*refs_son, *refs_mom], HF_TOKEN, distance_threshold=0.7)
    # print_summary(agg_results, "7 – Agglomerative (blind) - all sons + all moms")

    # # -------------------------------------------------------------------------
    # # Reference-guided (supervised / anchored) assignment runs
    # # -------------------------------------------------------------------------
    # print("\n" + "="*60)
    # print("REFERENCE-GUIDED ASSIGNMENT (supervised)")
    # print("="*60)

    # threshold = 0.65   # ← tune this value here (try 0.65, 0.68, 0.70, 0.72, 0.75)
    # # Central reference mapping
    # references = get_reference_mapping(refs_son, refs_mom)

    # print(f"\nRunning A - all sons (threshold = {threshold})")
    # results_a = run_reference_assignment(
    #     refs_son,
    #     HF_TOKEN,
    #     references,
    #     assignment_threshold=threshold,
    # )
    # print_summary(results_a, f"A – Reference-guided - all sons (thr={threshold})")

    # print(f"\nRunning B - all moms (threshold = {threshold})")
    # results_b = run_reference_assignment(
    #     refs_mom,
    #     HF_TOKEN,
    #     references,
    #     assignment_threshold=threshold,
    # )
    # print_summary(results_b, f"B – Reference-guided - all moms (thr={threshold})")

    # print(f"\nRunning C - all moms + 1 son (threshold = {threshold})")
    # results_c = run_reference_assignment(
    #     [*refs_mom, refs_son[0]],
    #     HF_TOKEN,
    #     references,
    #     assignment_threshold=threshold,
    # )
    # print_summary(results_c, f"C – Reference-guided - moms + 1 son (thr={threshold})")

    # print(f"\nRunning D - all sons + all moms (threshold = {threshold})")
    # results_d = run_reference_assignment(
    #     [*refs_son, *refs_mom],
    #     HF_TOKEN,
    #     references,
    #     assignment_threshold=threshold,
    # )
    # print_summary(results_d, f"D – Reference-guided - all sons + all moms (thr={threshold})")


if __name__ == "__main__":
    main()