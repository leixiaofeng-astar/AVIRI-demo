#!/bin/bash

[ "${-/i}" = "$-" ] && {
# added by Anaconda3 installer
export PATH="/home/user/anaconda3/bin:$PATH"

# The next line updates PATH for the Google Cloud SDK.
if [ -f '/home/user/google-cloud-sdk/path.bash.inc' ]; then . '/home/user/google-cloud-sdk/path.bash.inc'; fi

# The next line enables shell command completion for gcloud.
if [ -f '/home/user/google-cloud-sdk/completion.bash.inc' ]; then . '/home/user/google-cloud-sdk/completion.bash.inc'; fi
}

cd /home/user/project/xuxinx/VI

while :; do 
    python3 main_vi.py
done
