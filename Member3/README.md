# 🎯 Recommendation & User Modeling
## Overview

This module implements the personalization and recommendation layer of the tag-free AI music recommendation system. It operates entirely in learned embedding space, without using audio files, genres, or metadata. The goal is to model user musical preferences from listening behavior and recommend songs based on similarity in sound.

## What This Module Does

Builds a user preference vector from previously listened songs

Computes cosine similarity between user preferences and all songs

Ranks and recommends the most musically similar songs

Handles cold-start users with no listening history

Provides similarity scores to support explainable recommendations

This module is independent of feature extraction and embedding learning.

## Inputs
Input	              Description
song_embeddings	    NumPy array of shape (N, D) from Member 2
song_ids	          List of song identifiers (aligned with embeddings)
user_history	      List of indices of songs listened by the user

## Outputs

Ranked list of recommended songs

Similarity score for each recommendation

Song indices to support explainability and UI integration

## Core Logic
User Preference Modeling

User taste is represented as the mean of embeddings of listened songs and normalized for cosine similarity.

## Recommendation

Songs are ranked using cosine similarity between the user vector and all song embeddings. Previously listened songs are excluded.

## Cold Start

If insufficient user history is available, the system recommends musically representative songs based on embedding magnitude.

## Key Functions
normalize_embeddings()        # Ensures stable cosine similarity
build_user_vector()           # Builds normalized user preference vector
recommend_songs()             # Returns top-K personalized recommendations
cold_start_recommendation()   # Handles new users
explain_similarity()          # Supports explainable recommendations

## Design Decisions

Uses cosine similarity for scale-invariant comparison

Normalizes both song and user vectors

Keeps logic modular for easy UI integration

Supports explainability without exposing raw audio data

## Why This Matters

This module enables objective, tag-free personalization, ensuring fair music discovery and scalability to new or unlabeled songs. It converts learned musical understanding into actionable recommendations while remaining transparent and interpretable.
