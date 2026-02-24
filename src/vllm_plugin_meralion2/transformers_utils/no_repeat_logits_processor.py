"""No-repeat n-gram logits processor used by MERaLiON2 generation."""

from typing import Iterable, List

import torch


def _get_ngrams(ngram_size: int, prev_input_ids: torch.Tensor, num_hypos: int):
    """
    Assume ngram_size=2 and prev_input_ids=tensor([[40, 2883, 2712, 4346]]). The output of generated ngrams look like
    this {(40,): [2883], (2883,): [2712], (2712,): [4346]}.

    Args:
        ngram_size (`int`):
            The number sequential tokens taken as a group which may only occur once before being banned.
        prev_input_ids (`torch.Tensor`):
           Generated token ids for the current hypothesis.
        num_hypos (`int`):
            The number of hypotheses for which n-grams need to be generated.

    Returns:
        generated_ngrams (`dict`):
            Dictionary of generated ngrams.
    """
    # Initialize an empty list of dictionaries, one for each hypothesis (index) in the range of num_hypos
    generated_ngrams = [{} for _ in range(num_hypos)]
    for idx in range(num_hypos):
        gen_tokens = prev_input_ids[idx].tolist()
        generated_ngram = generated_ngrams[idx]
        # Loop through each n-gram of size ngram_size in the list of tokens (gen_tokens)
        for ngram in zip(*[gen_tokens[i:] for i in range(ngram_size)]):
            prev_ngram_tuple = tuple(ngram[:-1])
            generated_ngram[prev_ngram_tuple] = generated_ngram.get(
                prev_ngram_tuple, []
            ) + [ngram[-1]]
    return generated_ngrams


def _get_generated_ngrams(banned_ngrams, prev_input_ids, ngram_size, cur_len):
    """
    Determines the banned tokens for the current hypothesis based on previously generated n-grams.

    Args:
        banned_ngrams (`dict`):
            A dictionary containing previously generated n-grams for each hypothesis.
        prev_input_ids (`torch.Tensor`):
            Generated token ids for the current hypothesis.
        ngram_size (`int`):
            The number sequential tokens taken as a group which may only occur once before being banned.
        cur_len (`int`):
            The current length of the token sequences for which the n-grams are being checked.

    Returns:
        List of tokens that are banned.
    """
    # Before decoding the next token, prevent decoding of ngrams that have already appeared
    start_idx = cur_len + 1 - ngram_size
    ngram_idx = tuple(prev_input_ids[start_idx:cur_len].tolist())
    return banned_ngrams.get(ngram_idx, [])


def _calc_banned_ngram_tokens(
    ngram_size: int, prev_input_ids: torch.Tensor, num_hypos: int, cur_len: int
) -> List[Iterable[int]]:
    """Copied from fairseq for no_repeat_ngram in beam_search"""
    if cur_len + 1 < ngram_size:
        # return no banned tokens if we haven't generated no_repeat_ngram_size tokens yet
        return [[] for _ in range(num_hypos)]
    generated_ngrams = _get_ngrams(ngram_size, prev_input_ids, num_hypos)
    banned_tokens = [
        _get_generated_ngrams(
            generated_ngrams[hypo_idx], prev_input_ids[hypo_idx], ngram_size, cur_len
        )
        for hypo_idx in range(num_hypos)
    ]
    return banned_tokens


class NoRepeatNGramLogitsProcessor:
    """Logits processor that prevents repetition of n-grams.

    This processor sets the logits of tokens that would create repeated
    n-grams to negative infinity, effectively preventing them from being
    selected during sampling.
    """

    def __init__(self, ngram_size: int = 6) -> None:
        """Initialize the no-repeat n-gram processor.

        Args:
            ngram_size: Size of n-grams to prevent from repeating.
                Must be a positive integer.

        Raises:
            ValueError: If ngram_size is not a positive integer.
        """
        if not isinstance(ngram_size, int) or ngram_size <= 0:
            raise ValueError(
                f"`ngram_size` has to be a strictly positive integer, "
                f"but is {ngram_size}"
            )
        self.ngram_size = ngram_size

    def __call__(
        self,
        prompt_tokens_ids: tuple,
        past_tokens_ids: tuple,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """Process logits to prevent n-gram repetition.

        Args:
            prompt_tokens_ids: Tuple of prompt token IDs.
            past_tokens_ids: Tuple of past generated token IDs.
            scores: Logits tensor of shape [B, vocab_size] or [vocab_size].

        Returns:
            Processed logits tensor with repeated n-gram tokens set to -inf.
        """

        input_ids = prompt_tokens_ids + past_tokens_ids
        if len(input_ids) < self.ngram_size:
            return scores

        if len(scores.shape) == 1:
            scores = scores.reshape(1, -1)

        num_batch_hypotheses = scores.shape[0]

        input_ids_tensor = torch.LongTensor(input_ids).reshape(
            num_batch_hypotheses, -1
        )
        cur_len = input_ids_tensor.shape[-1]

        scores_processed = scores.clone()
        banned_batch_tokens = _calc_banned_ngram_tokens(
            self.ngram_size,
            input_ids_tensor,
            num_batch_hypotheses,
            cur_len,
        )

        for i, banned_tokens in enumerate(banned_batch_tokens):
            scores_processed[i, banned_tokens] = -float("inf")

        return scores_processed
