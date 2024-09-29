import torch
import torchaudio
import warnings

"""
Fast Griffin-Lim Algorithm with custom initialization option, using PyTorch
Ported from Librosa implementation.
"""

# Fast Griffin-Lim algorithm PyTorch implementation (Tested: OK)
def FGLA_custom_torch(S,
                      n_iter=32,
                      hop_length=64, 
                      win_length=512,
                      n_fft=512,
                      window="hann",
                      center=True,
                      dtype=None,
                      length=None,
                      pad_mode="constant",
                      momentum=0.99,
                      init="None",
                      init_tensor=None,  # initial phase estimate [torch.FloatTensor]
                      random_state=None):

  if momentum > 1:
    warnings.warn(
      "Griffin-Lim with momentum={} > 1 can be unstable. "
      "Proceed with caution!".format(momentum),
      stacklevel=2,
    )
  elif momentum < 0:                   
    raise ValueError("griffinlim() called with momentum={} < 0".format(momentum))
		 
  # Infer n_fft from the spectrogram shape
  if n_fft is None:
    n_fft = 2 * (S.shape[-2] - 1)

  # using complex64 will keep the result to minimal necessary precision
  angles = torch.empty(S.shape, dtype=torch.complex64)
  eps = torch.tensor(1e-8, dtype=S.dtype)

  if init == "random":
    # randomly initialize the phase
    angles[:] = torch.exp(2j * torch.tensor([3.14159265]) * torch.rand(*S.shape))

  # added new init option
  elif init == "custom":				
    angles[:] = torch.exp(2j * torch.tensor([3.14159265]) * init_tensor)
                
  elif init is None:
    # Initialize an all ones complex matrix
    angles[:] = 1.0

  else:
    raise ValueError("init={} must be 'custom' or 'random' or None".format(init))

  # And initialize the previous iterate to 0
  rebuilt = torch.zeros_like(S)			  

  for _ in range(n_iter):

    # Store the previous iterate
    tprev = rebuilt

    # Invert with our current estimate of the phases
    inverse = torchaudio.functional.inverse_spectrogram(S * angles,
                                                        n_fft=n_fft,
                                                        hop_length=hop_length,
                                                        win_length=win_length,
                                                        window=torch.hann_window(win_length),
                                                        pad=0,
                                                        center=center,
                                                        pad_mode=pad_mode,
                                                        length=length,
                                                        normalized=True,
                                                        )

    # time -> time-frequency
    rebuilt = torchaudio.functional.spectrogram(inverse,
                                                n_fft=n_fft,
                                                hop_length=hop_length,
                                                win_length=win_length,
                                                window=torch.hann_window(win_length),
                                                pad=0,
                                                power=None,
                                                normalized=True,
                                                center=center,
                                                pad_mode=pad_mode,
                                                )

    # Update phase estimates
    angles[:] = rebuilt - (momentum / (1 + momentum)) * tprev
    angles[:] /= torch.abs(angles) + eps

  # Return the final phase estimates and metrics
  return inverse