import matplotlib.pyplot as plt
from pathlib import Path
from muse_toolbox.data.databases.ami import AMIDatabase
from muse_toolbox.data.databases.ami_label_parsers import AMISegmentsXMLParser

def test_parser():
    data_dir = "/data4/Henri/MuSE-Toolbox/data/databases/ami"
    fs = 16000
    parser = AMISegmentsXMLParser(data_dir=data_dir, sampling_frequency=fs)
    db = AMIDatabase(data_dir=data_dir, label_parser=parser, sampling_frequency=fs)
    
    print("Testing AMIDatabase with meeting EN2001a and Array1")
    audio, sad_samples = db.get_meeting("EN2001a", "Array1")
    print(f"Audio shape: {audio.shape}")
    
    # Plot the first 1 minute of activity
    plt.figure(figsize=(10, 4))
    for i, (spk, act) in enumerate(sad_samples.items()):
        act_np = act[:fs*60].numpy()
        plt.plot(act_np + i*1.2, label=f"Speaker {spk}")
    
    plt.title("Speaker Activity (First 60s)")
    plt.yticks([])
    plt.legend()
    
    out_path = Path("/home/henrig/.gemini/antigravity-ide/brain/52fea5db-8f00-4d55-b293-ad99d1c95434/sad_plot.png")
    plt.savefig(out_path)
    print(f"Plot saved to {out_path}")

if __name__ == "__main__":
    test_parser()
