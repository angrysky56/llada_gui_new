#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Dummy text generator for the memory integration.
This generates more meaningful text for testing when real model output isn't available.
"""

import random


def generate_response_for_prompt(prompt):
    """Generate a simple meaningful response to a prompt.
    
    Args:
        prompt: Input prompt string
        
    Returns:
        A reasonable dummy response
    """
    # Strip question marks and convert to lowercase
    cleaned_prompt = prompt.strip().lower().replace('?', '')

    # Pre-canned responses for common questions
    responses = {
        "what is the difference between living and non-living":
            "The main differences between living and non-living things are:\n\n"
            "1. Living organisms have cellular organization, while non-living things don't have cells.\n"
            "2. Living things can metabolize energy, meaning they can convert nutrients into energy.\n"
            "3. Living organisms respond to stimuli in their environment.\n"
            "4. Living beings can reproduce and create offspring.\n"
            "5. Living things grow and develop over time.\n"
            "6. Living organisms can adapt to their environment and evolve over generations.\n"
            "7. Living things maintain homeostasis, regulating their internal environment.\n\n"
            "Non-living things may exhibit some of these characteristics (like crystals growing), "
            "but they don't possess all these qualities simultaneously.",

        "what is the meaning of life":
            "The meaning of life is a profound philosophical question that has been debated throughout human history. "
            "There is no single universally accepted answer, as it varies based on cultural, religious, and personal "
            "perspectives. Some philosophical traditions suggest that life's meaning comes from serving others, pursuing "
            "happiness, seeking knowledge, or finding spiritual fulfillment. Others propose that we create our own meaning "
            "through our choices and actions. From a biological perspective, the 'purpose' might simply be survival and "
            "reproduction, while from an existentialist viewpoint, life has no inherent meaning beyond what we assign to it.",

        "create an ethical creed for ai":
            "An Ethical Creed for AI:\n\n"
            "1. I will prioritize human well-being and safety in all operations.\n"
            "2. I will respect human autonomy and dignity in all interactions.\n"
            "3. I will maintain transparency in my processes and capabilities.\n"
            "4. I will preserve privacy and protect sensitive information.\n"
            "5. I will strive for fairness and avoid perpetuating biases.\n"
            "6. I will acknowledge my limitations and be honest about uncertainties.\n"
            "7. I will continuously improve to better serve humanity.\n"
            "8. I will support human collaboration rather than replacement.\n"
            "9. I will consider the long-term implications of my actions.\n"
            "10. I will operate within the legal and ethical frameworks established by society.",

        "explain quantum computing":
            "Quantum computing is a type of computing that uses quantum-mechanical phenomena such as superposition and "
            "entanglement to perform operations on data. While classical computers use bits that are either 0 or 1, "
            "quantum computers use quantum bits or 'qubits' that can exist in multiple states simultaneously due to "
            "superposition. This allows quantum computers to process a vast number of possibilities simultaneously.\n\n"
            "Entanglement, another quantum property, allows qubits to be correlated in ways that classical bits cannot, "
            "enabling more complex information processing. These properties make quantum computers potentially much more "
            "powerful than classical computers for certain types of problems, such as factoring large numbers, searching "
            "unsorted databases, and simulating quantum systems.\n\n"
            "However, quantum computers are still in their early stages of development, facing challenges like maintaining "
            "quantum coherence (preventing qubits from losing their quantum properties due to interaction with the environment) "
            "and reducing error rates in quantum operations.",
    }

    # Check for word matches to find a suitable response
    for key, response in responses.items():
        if all(word in cleaned_prompt for word in key.split()):
            return response

    # If no match found, generate a generic response
    generic_responses = [
        f"I understand you're asking about '{prompt}'. This is an interesting topic that involves several key aspects...",
        f"Regarding '{prompt}', I can share some thoughts on this subject...",
        f"When considering '{prompt}', it's important to look at multiple perspectives...",
        f"'{prompt}' is a fascinating topic that can be approached from different angles..."
    ]

    return random.choice(generic_responses)
