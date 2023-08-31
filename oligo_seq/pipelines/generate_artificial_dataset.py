import os
import shutil
import argparse
import yaml
from random import sample
import copy
import time
from typing import Tuple

from oligo_designer_toolsuite.database import NcbiGenomicRegionGenerator, EnsemblGenomicRegionGenerator, CustomGenomicRegionGenerator, OligoDatabase, ReferenceDatabase
from oligo_designer_toolsuite.oligo_property_filter import PropertyFilter, MaskedSequences
from oligo_designer_toolsuite.oligo_specificity_filter import SpecificityFilter, ExactMatches
from Bio.Seq import MutableSeq, Seq
from Bio.SeqUtils import gc_fraction
from Bio.SeqUtils import MeltingTemp as mt
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import nupack
from tqdm import tqdm
from math import log
import joblib


base_pair = {'A':'T', 'T':'A', 'C':'G', 'G':'C'} #, 'a':'t', 't':'a', 'c':'g', 'g':'c'}


def reverse_complement(strand: str) -> str:
    strand = list(strand)
    strand.reverse()
    for i in range(len(strand)):
        strand[i] = base_pair[strand[i]]
    return "".join(strand)


def mutate(nt: str) -> str:
    nts = ['A', 'C', 'T', 'G']
    nts.remove(nt)
    return sample(nts, 1)[0]


def duplexing_log_scores(oligo: str, off_target: str, model: nupack.Model, concentration: float) -> float:
     # oligo must be reversed and complemented
    oligo_strand = nupack.Strand(oligo, name="oligo")
    on_target_strand = nupack.Strand(reverse_complement(oligo), name="on_target")
    off_target_strand = nupack.Strand(reverse_complement(off_target), name="off_target")
    t = nupack.Tube(strands={oligo_strand: concentration, on_target_strand: concentration, off_target_strand: concentration}, name='t', complexes=nupack.SetSpec(max_size=2))
    tube_results = nupack.tube_analysis(tubes=[t], model=model)
    tube_concentrations = tube_results[t].complex_concentrations
    # d_score.append((tube_concentrations[nupack.Complex(strands=[oligo,on_target])] / (tube_concentrations[nupack.Complex(strands=[oligo,off_target])]))) # foldchange on target vs off target concetration
    return log(tube_concentrations[nupack.Complex(strands=[oligo_strand,on_target_strand])] / (tube_concentrations[nupack.Complex(strands=[oligo_strand,off_target_strand])])) # logfoldchange on target vs off target concetration


def generate_off_targets(sequence: Seq, config) -> list[Tuple[str,str, int, float]]:
    model = nupack.Model()
    off_target_regions = [(str(sequence), str(sequence), 0, duplexing_log_scores(str(sequence), str(sequence), model, config["concentration"]))] # include an exact match
    #mutations
    for i in range(config["n_mutations"]): # nr of mutations
        for j in range(len(sequence)):
            # mutate j and other i-1 nt
            target = MutableSeq(sequence)
            new_nt = mutate(target[j])
            target.pop(j)
            target.insert(j, new_nt)
            unchanges_nts = list(range(len(sequence)))
            unchanges_nts.remove(j)
            for _ in range(i-1):
                k = sample(unchanges_nts, 1)[0]
                new_nt = mutate(target[k])
                target.pop(k)
                target.insert(k, new_nt)
                unchanges_nts.remove(k)
            off_target_regions.append((str(sequence), str(target), i+1, duplexing_log_scores(str(sequence), str(target), model, config["concentration"])))
    return off_target_regions


def main():
    """Generate an artificial dataset containing oligos and some hand-crafted mutations with the 
    relative mutations scores. The oligos are extracted form a given list of genes and uniformly sampled to match 
    the desidred dataset size. These oligos are then mutated by applying 0 to max_mutaions base-pairs mutations to generate potential off-targets.
    (REMARK: for each nr. of mutations we create an off-target region startic from each nucleotide of the oligo sequence
    and selecting the remaining mutated nucleotides uniformly. Therefore, from each oligo we generate 
    O(max_mutations * oligo_length) off-target regions.)

    The duplexing score is obtained from the final concentration of DNA complexes in NUPACK tube experiment simulation
    that contains the oligo sequence, the exact on-target region and the off-target. The oligo, on-target and off-target
     strands are initially set at the same concentration $C_{in}$ and we define the duplexing score as: 
    
    log( C_{oligo + off-t} /C_{in} + eps ). 
    
    The oligos, the on-target regions and off-target regions are inserted in order to compare the amount of oligos that 
    bind to one and to the other. Additionally the log is used to sterch the scored distribution making them 
    easier to predict and a small value eps = 1e-12 is used for numerical stability.
    """

    #########################
    # read in out arguments #
    #########################

    start = time.time()
    parser = argparse.ArgumentParser(
        prog="Artificial Dataset",
        usage="generate_artificial_dataset [options]",
        description=main.__doc__,
    )
    parser.add_argument("-c", "--config", help="path to the configuration file", default="config/generate_artificial_dataset.yaml")
    args = parser.parse_args()
    with open(args.config, "r") as handle:
        config = yaml.safe_load(handle)

    ################################
    # generate the oligo sequences #
    ################################

    if config["annotation_source"] == "ncbi":
        # dowload the fasta files formthe NCBI server
        region_generator = NcbiGenomicRegionGenerator(
            taxon=config["taxon"],
            species=config["species"],
            annotation_release=config["annotation_release"],
            dir_output="output",
        )
    elif config["annotation_source"] == "ensembl":
        # dowload the fasta files formthe NCBI server
        region_generator = EnsemblGenomicRegionGenerator(
            species=config["species"],
            annotation_release=config["annotation_release"],
            dir_output="output",
        )
    elif config["annotation_source"] == "custom":
        # use already dowloaded files
        region_generator = CustomGenomicRegionGenerator(
            annotation_file=config["file_annotation"],
            sequence_file=config["file_sequence"],
            files_source=config["files_source"],
            species=config["species"],
            annotation_release=config["annotation_release"],
            genome_assembly=config["genome_assembly"],
            dir_output="output",
        )
    file_transcriptome = region_generator.generate_transcript_reduced_representation()
    # oligo database
    oligo_database = OligoDatabase(
        file_fasta=file_transcriptome,
        files_source=region_generator.files_source,
        species=region_generator.species,
        annotation_release=region_generator.annotation_release,
        genome_assembly=region_generator.genome_assembly,
        n_jobs=config["n_jobs"],
        dir_output="output",
    )
    with open(config["file_genes"]) as handle:
        lines = handle.readlines()
        genes = [line.rstrip() for line in lines]
    oligo_database.create_database(
        oligo_length_min = config["min_length"], 
        oligo_length_max = config["max_length"], 
        region_ids = genes
    )
    # Property filtering
    masked_seqeunces = MaskedSequences()
    property_filter = PropertyFilter(filters=[masked_seqeunces])
    oligo_database = property_filter.apply(oligo_database=oligo_database, n_jobs=4)
    # Specificity filtering
    reference_database = ReferenceDatabase(
        file_fasta=file_transcriptome, # or just the transciptome?
        files_source="NCBI", 
        species="Human", 
        annotation_release="110", 
        genome_assembly="GRCh38",
        dir_output="output_artificial",
    )
    exact_matches = ExactMatches(dir_specificity="specificity")
    specificity_filter = SpecificityFilter(filters=[exact_matches])
    oligo_database = specificity_filter.apply(oligo_database=oligo_database, reference_database=reference_database)
    print("Oligo seqeunces generated.")

    ##############################
    # sample the oligo sequences #
    ##############################

    gc_content =[]
    oligo_length = []
    # distribution of the GC content and length
    for gene, oligos in oligo_database.database.items():
        for oligo, features in oligos.items():
            gc_content.append([gc_fraction(features["sequence"]), "Initial"])
            oligo_length.append([features["length"], "Initial" ])
    # sample the oligos
    sample_per_gene = round(config["n_sampled"]/len(genes)) # sample the same number of oligos from each gene
    sampled_oligos = {}
    for gene, oligos in oligo_database.database.items():
        sampled_oligos[gene] = {}
        # ids of the samples oligos
        ids = sample(population=list(oligos.keys()), k=min(sample_per_gene, len(oligos)))
        for id in ids:
            sampled_oligos[gene][id] = copy.deepcopy(oligos[id])
            gc_content.append([gc_fraction(sampled_oligos[gene][id]["sequence"]), "Sampled"])
            oligo_length.append([sampled_oligos[gene][id]["length"], "Sampled" ])
    gc_content = pd.DataFrame(data=gc_content, columns=["GC content", "State"])
    oligo_length = pd.DataFrame(data=oligo_length, columns=["Length", "State"])
    oligo_database.database = sampled_oligos
    plt.figure(1)
    sns.violinplot(data=gc_content, y="GC content", x="State")
    plt.title("GC content distribution")
    plt.figure(2)
    sns.violinplot(data=oligo_length, y="Length", x="State")
    plt.title("Oligo length distribution")
    print("Oligo sequences sampled.")

  
    ################################################################
    # generate artificial off-targets and compute duplexing scores #
    ################################################################
    

    alignments = joblib.Parallel(n_jobs=config["n_jobs"])(
        joblib.delayed(generate_off_targets)(
            oligo_features["sequence"].upper(), config
        )
        for _, oligos in oligo_database.database.items() for _, oligo_features in oligos.items()
    )
    alignments = [alignment for oliigo_alignments in alignments for alignment in oliigo_alignments]


    ##################
    # create dataset #
    ##################


    dataset = pd.DataFrame(index=list(range(len(alignments))), columns=["oligo_sequence", "oligo_length", "oligo_GC_content", "off_target_sequence", "off_target_legth", "off_target_GC_content", "tm_diff", "number_mismatches", "duplexing_log_score"])
    for i, (oligo, off_target, nr_mismatches, d_log_score) in enumerate(alignments):
        dataset.loc[i] = [
            oligo, #oligo sequence
            len(oligo),# oligo length
            gc_fraction(oligo),
            off_target,
            len(off_target), # off target length
            round(gc_fraction(off_target)), # off target gc content
            mt.Tm_NN(oligo) - mt.Tm_NN(off_target),
            nr_mismatches,
            d_log_score,
        ]
    plt.figure(3)
    sns.boxplot(x=dataset["duplexing_log_score"])
    plt.title("Duplexig scores distribution")
    os.makedirs(config["output"], exist_ok=True)
    file_dataset = os.path.join(config["output"], f"artificial_dataset_{config['min_length']}_{config['max_length']}.csv")
    dataset.to_csv(file_dataset)
    print(f"Dataset created and stored at {file_dataset}.")
    print(f"Computational time: {time.time() - start}")
    shutil.rmtree("output")
    plt.show()


if __name__ == "__main__":
    main()