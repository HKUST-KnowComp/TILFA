"""
=========================================================
Title: ImageArg Shared Task Code - Dataset Downloader
---------------------------------------------------------
Warning: The downloaded dataset should only be used for
participating in ImageArg Shared Task. Any other use is
explicitly prohibited. Any participants are not allowed
to redistribute the dataset per Twitter Developer Policy:
https://developer.twitter.com/en/developer-terms/policy.
---------------------------------------------------------
Notice: This code is managed by ImageArg Shared Task
(https://imagearg.github.io/).
---------------------------------------------------------
Data: 2023-07-08
=========================================================
"""
import platform

if __name__ == '__main__':
    if platform.processor() == "arm":
        from plugins.macos_silicon import get_test_data_without_labels
    elif platform.system() == "Darwin" and platform.machine() == "x86_64":
        from plugins.macos_intel import get_test_data_without_labels
    elif platform.system() == "Linux" and platform.machine() == "x86_64":
        from plugins.ubuntu import get_test_data_without_labels
    else:
        raise "not supported system!"
    get_test_data_without_labels()


