#!/bin/bash


# This is a test script for both `memory leaks` and correct outputs (corresponding to the output files that Yuval created), 
# as well as an efficiency test, for those who don't care to optimize their lanky code.
# You're welcome.
# - Adam
#
# For help:
# bash tester.sh --help




# global variables
testers_path=$1
leaks=$2
output_file="./tmp/output.txt"
valgrind_file="./tmp/valgrind.txt"
readarray -t k_array < $1/k_array.txt

# buffering between tests
function buffer() {
	echo -e "\n"
	echo "
*****
*   *
*   *
*   *
*****"
	echo -e "\n"
}



# =================
# REGULAR TEST
# =================
function regular_test() {

	if [[ $interface == @(c|both) ]]; then
		# testing the C interface
		rm $results_dir/test_transcript_c.txt &> /dev/null
		rm $results_dir/memory_transcript_c.txt &> /dev/null
		touch $results_dir/test_transcript_c.txt
		touch $results_dir/memory_transcript_c.txt
		echo -e "\n\e[4;37mTesting correct outputs for the interface of \e[4;33m\e[1;33mC\e[0m:"
		echo -e "\n\e[4;34m\e[1;34mRESULTS\e[0m"
		test_interface c
		
		buffer
	fi


	if [[ $interface == @(py|both) ]]; then
		
		# If on nova, no pint on checking memory leaks, if regular="no" noting left to do in regular_test
		valgrindversion=$(valgrind --version)
		if [[ $valgrindversion == "valgrind-3.10.0.SVN" ]]; then
			echo -e "Valgrind on NOVA can't detect memory leaks"
			leaks="no"
			if [[ $regular == "no" ]]; then
				return
			fi
		fi
	
		# testing the CPython interface
		rm $results_dir/test_transcript_py.txt &> /dev/null
		rm $results_dir/memory_transcript_py.txt &> /dev/null
		touch $results_dir/test_transcript_py.txt
		touch $results_dir/memory_transcript_py.txt
		cp $testers_path/valgrind-python.supp valgrind-python.supp
		echo -e "\e[4;37mTesting correct outputs for the interface of \e[4;33m\e[1;33mPython\e[0m:"
		echo -e "\n\e[4;34m\e[1;34mRESULTS\e[0m"
		test_interface py
		
		buffer
	fi

}


# comprehensive interface test - specific interface only
function test_interface() {
	# the first argument shall be the interface being tested
	local_interface=$1

	# else, run a test
	test_goal $local_interface wam
	test_goal $local_interface ddg
	test_goal $local_interface lnorm
	test_goal $local_interface jacobi

	if [[ $local_interface == "py" ]]; then
		test_goal $local_interface spk
	fi
}



# comprehensive goal test - specific goal, specific interface only
function test_goal() {
	# the first argument shall be the interface being tested c/py
	# the second argument shall be the goal being tested

	# running a test for the goal
	if [[ "${2}" == "jacobi" ]]; then
		for (( i = 0; i <= $jacobi; i++ )); do
			echo -n "${1^^}: ${2^^}: ${testers_path}/jacobi_${i}.txt: "
			individual_test $1 $2 jacobi_$i.txt 0
			echo
		done
	else
		for (( i = 0; i <= $spk; i++ )); do
			echo -n "${1^^}: ${2^^}: ${testers_path}/spk_${i}.txt: "
			individual_test $1 $2 spk_$i.txt $i
			echo
		done
	fi
	


}




# invidividual test - specific input file, specific goal, specific interface only
function individual_test() {
	# the first argument shall be the interface being tested: c/py
	# the second argument shall be the goal being tested
	# the third argument shall be the input file being used
	# the fourth argument shall be the input file index

	# running the commands
	if [[ "${1}" == "py" ]]; then # if we are testing the python interface
		if [[ $leaks == "yes" ]]; then
			valgrind --leak-check=full --show-leak-kinds=definite --log-file=$valgrind_file --suppressions=valgrind-python.supp python3.8-dbg -E spkmeans.py ${k_array[$4]} $2 $testers_path/$3 &> $output_file
		else
			python3 -E spkmeans.py ${k_array[$4]} $2 $testers_path/$3 &> $output_file
		fi
	elif [[ "${1}" == "c" ]]; then # if we are testing the C interface
		valgrind --leak-check=full --show-leak-kinds=all --log-file=$valgrind_file ./spkmeans $2 $testers_path/$3 &> $output_file
	else
		echo "Individual test function failed: Invalid interface"
		return -1
	fi

	# if a regular test was asked
	if [[ $regular == "yes" ]]; then
		# calculating the difference between the desired output and the actual output
		diff_result=$(diff $output_file $testers_path/outputs/$1/$2/$3 2>&1)
		
		# verdicting if the test failed, then print an appropriate status
		verdict_diff ${#diff_result}
		
		# if the test failed, print a report of the 'diff' operation into the test transcript of the interface
		if [[ ${#diff_result} -ne 0 ]]; then 
			echo -e "DIFF RESULT FOR: ${1}: ${2}: ${3}:\n${diff_result}\n\n" >> $results_dir/test_transcript_$1.txt
		fi
	fi
	
	# verdict if the test had a memory leak, and print an appropriate status accordingly
	if [[ $leaks == "yes" ]]; then
		if [[ $1 == "c" ]]; then
		
			# buffering
			if [[ $regular == "yes" ]]; then
				echo -n ": "
			fi
			
			# running memory test
			verdict_memory_loss_c
			echo -e "MEMORY LEAK RESULT FOR: ${1}: ${2}: ${3}:\n\n" >> $results_dir/memory_transcript_c.txt
			cat $valgrind_file >> $results_dir/memory_transcript_c.txt
			echo -e "\n\n\n" >> $results_dir/memory_transcript_c.txt
		fi

		if [[ $1 == "py" ]]; then
		
			# buffering
			if [[ $regular == "yes" ]]; then
				echo -n ": "
			fi
			
			# running memory test
			verdict_memory_loss_py
			echo -e "MEMORY LEAK RESULT FOR: ${1}: ${2}: ${3}:\n\n" >> $results_dir/memory_transcript_py.txt
			cat $valgrind_file >> $results_dir/memory_transcript_py.txt
			echo -e "\n\n\n" >> $results_dir/memory_transcript_py.txt
		fi
	fi
}



function verdict_diff() {
	# the first argument shall be the length (in characters) of the result of diff
	
	if [[ $1 -eq 0 ]]; then
		echo -ne '\033[1;32mSUCCESS\e[0m' # print out a 'success' message
	else
		echo -ne "\e[1;31mFAILED\e[0m" # print out a 'failed' message
	fi
}






function verdict_memory_loss_c() {
	# this function shall be used only upon the C interface comprehensive test
	# this function assumes the $valgrind_file matches the very last running of a test using the C interface
	
	bytes_lost=$(cat $valgrind_file | grep "HEAP SUMMARY" -A 1 | tail -n1 | awk '{print $6}')
	
	if [[ ${bytes_lost//,} -eq 0 ]]; then
		echo -ne '\033[1;32mNO MEMORY LEAK\e[0m' # print out a 'success' message
	else
		echo -ne "\e[1;31mMEMORY LEAK\e[0m" # print out a 'failed' message
	fi
}

function verdict_memory_loss_py() {
	# this function shall be used only upon the Py interface comprehensive test
	# this function assumes the $valgrind_file matches the very last running of a test using the Py interface
	
	if grep -q "LEAK SUMMARY" $valgrind_file; then
		bytes_lost=$(cat $valgrind_file | grep "LEAK SUMMARY" -A 1 | tail -n1 | awk '{print $4}')
		
		if [[ ${bytes_lost//,} -eq 0 ]]; then
			echo -ne '\033[1;32mNO MEMORY LEAK\e[0m' # print out a 'success' message
		else
			echo -ne "\e[1;31mMEMORY LEAK\e[0m" # print out a 'failed' message
		fi
	else
    	echo -ne "\e[1;31mCouldn't Find LEAK SUMMARY in Valgrind output, can't know if there is a leak\e[0m" # print out a 'failed' message
	fi
}





# =================
# EFFICIENCY TEST
# =================
function efficiency_test() {
	# test message
	echo -e "\e[4;37mTesting the \e[4;33m\e[1;33mefficiency of the algorithms\e[0m.\n\e[1;31mNOTICE: This test is going to take quite a while.\nMoreover, this test is subjective.\nAlbeit, don't be a bozo and do optimize your code - you might get a penalty for inefficient code.\e[0m"
	
	# creating a maximized file
	echo -e 'from sklearn import datasets
import numpy as np
def save_dataset(dataset, filename):
    lines = map(stringify_row, dataset)
    with open(filename + ".csv", "w") as f:
        f.writelines(lines)
     
def stringify_row(row):
    row = map(str, row)
    return ",".join(row) + "\\n"

# datapoints      
save_dataset(datasets.make_blobs(n_samples=1000, centers=3, cluster_std=5, n_features=10, shuffle=True, random_state=31)[0], "1000_blobs_10_feat")

# symmetric matrix
a = np.random.rand(1000, 1000)
m = np.tril(a) + np.tril(a, -1).T
save_dataset(m, "jacobi_input_10_6")' | python3
	
	if [[ $interface == @(c|both) ]]; then
		# testing the efficiency of the C interface
		rm $results_dir/efficiency_transcript_c.txt &> /dev/null
		touch $results_dir/efficiency_transcript_c.txt
		echo -e "\n\e[4;37mTesting the efficiency for the interface: \e[4;33m\e[1;33mC\e[0m:"
		echo -e "\n\e[4;34m\e[1;34mRESULTS\e[0m"
		test_efficiency_interface c
		
		buffer
	fi
	
	
	if [[ $interface == @(py|both) ]]; then
		# testing the efficiency of the CPython interface
		rm $results_dir/efficiency_transcript_py.txt &> /dev/null
		touch $results_dir/efficiency_transcript_py.txt
		echo -e "\e[4;37mTesting the efficiency for the interface: \e[4;33m\e[1;33mPython\e[0m:"
		echo -e "\n\e[4;34m\e[1;34mRESULTS\e[0m"
		test_efficiency_interface py
		
		buffer
	fi
	
	rm jacobi_input_10_6.csv &> /dev/null
	rm 1000_blobs_10_feat.csv &> /dev/null
}


function test_efficiency_interface() {
	# the first argument shall be the interface being tested
	local_interface=$1

	test_efficiency_goal $local_interface wam
	test_efficiency_goal $local_interface ddg
	test_efficiency_goal $local_interface lnorm
	test_efficiency_goal $local_interface jacobi

	if [[ $local_interface == "py" ]]; then
		test_efficiency_goal $local_interface spk
	fi
}


function test_efficiency_goal() {
	# the first argument is the interface that is currently being tested
	# the second argument is the goal being tested
	
	echo -n "${1^^}: ${2^^}: Checking efficiency: "
	local time_result
	
	if [[ $1 == "c" ]]; then
	
		if [[ $2 == "jacobi" ]]; then
			time_result=$(timeout 11.5 bash -c "time ./spkmeans jacobi jacobi_input_10_6.csv 1> /dev/null" 2>&1) 
		else
			time_result=$(timeout 0.65 bash -c "time ./spkmeans ${2} 1000_blobs_10_feat.csv 1> /dev/null" 2>&1) 
		fi
	
	else
		if [[ $2 == "jacobi" ]]; then
			time_result=$(timeout 8 bash -c "time python3 spkmeans.py 0 jacobi jacobi_input_10_6.csv 1> /dev/null" 2>&1) 
		elif [[ $2 == "spk" ]]; then
			time_result=$(timeout 130 bash -c "time python3 spkmeans.py 0 spk 1000_blobs_10_feat.csv 1> /dev/null" 2>&1) 
		else
			time_result=$(timeout 4.5 bash -c "time python3 spkmeans.py 0 ${2} 1000_blobs_10_feat.csv 1> /dev/null" 2>&1) 
		fi
	fi
	
	
	# verdicting result of test, and outputting a corresponding log message to the efficiency transcript
	if [[ ${#time_result} -ne 0 ]]; then # test succeeded
		echo -e '\033[1;32mSUCCESS\e[0m' # print out a 'success' message
	else
		echo -e "\e[1;31mFAILED\e[0m" # print out a 'failed' message
		echo -e "REQUIRED EFFICIENCY FOR: ${1}: ${2}: is:\n" >> $results_dir/efficiency_transcript_$1.txt
		
		if [[ $1 == "c" ]]; then
			if [[ $2 == "jacobi" ]]; then
				echo -ne "11.5" >> $results_dir/efficiency_transcript_$1.txt
			else
				echo -ne "0.65" >> $results_dir/efficiency_transcript_$1.txt
			fi
		else
			if [[ $2 == "jacobi" ]]; then
				echo -ne "8" >> $results_dir/efficiency_transcript_$1.txt
			elif [[ $2 == "spk" ]]; then
				echo -ne "130" >> $results_dir/efficiency_transcript_$1.txt
			else
				echo -ne "4.5" >> $results_dir/efficiency_transcript_$1.txt
			fi
		fi
		
		echo -e " seconds.\nGood luck next time!\n\n\n" >> $results_dir/efficiency_transcript_$1.txt
	fi
	
	
}





# =================
# Organizer
# =================
function comprehensive_test() {

	mkdir ./tmp &> /dev/null

	# Trying to build the necessary resources
	if [[ $interface == @(c|both) ]]; then
		comp_output=$(bash comp.sh 2>&1) # compiling
		if [[ ${#comp_output} -ne 0 ]]; then
			echo -e "\e[1;31mFailed to compile your C module with \`\033[4;31mcomp.sh\e[0m\e[1;31m\`!\n"
			youre_a_bozo
		fi
	fi
	
	if [[ $interface == @(py|both) ]]; then
		build_output=$(python3 setup.py build_ext --inplace 2>&1 1>/dev/null)
		if [[ ${#build_output} -ne 0 ]]; then
			echo -e "\e[1;31mFailed to build the CPython extension with \`\033[4;31msetup.py\e[0m\e[1;31m\`!\n"
			youre_a_bozo
		fi
	fi
	
	# Running the necessary tests
	if [[ $regular == "yes" || $leaks == "yes" ]]; then
		regular_test
	fi
	
	if [[ $efficiency == "yes" ]]; then
		efficiency_test
	fi
	
	
	# Summary 
	echo -e "\033[4;31m\e[1;31mDONE -> NOTICE:\e[0m\nDetailed regular tests' results are in: \e[1;34m${results_dir}/test_transcript_[c|py].txt\e[0m.\nDetailed memory leak tests' results have their memory reports at \e[1;34m${results_dir}/memory_transcript_[c|py].txt\e[0m.\nDetailed efficiency tests' results have their efficiency reports at \e[1;34m${results_dir}/efficiency_transcript_[c|py].txt\e[0m.\n\e[4;37mOnly the results of failed tests will be viewed in the transcripts.\e[0m"
}





# =================
# PRELUDE
# =================
function instructions() {
	echo -e "\e[4;37m\e[1;37mHelp\e[0m: bash tester.sh <testfiles> <interface> <regular> <leaks> <efficiency> [results_dir]\n
- testfiles	<path>: 	The path of the directory containing the test files that were supplied with this test.
- interface 	[c|py|both]:	Specifies the interface you want to test.
- regular 	[yes|no]:	Specifies whether to run the regular tests or not.
- leaks		[yes|no]:	Specifies if you want memory leak tests.
- efficiency	[yes|no]:	Specifies whether you want an efficiency test on the interfaces you chose.
- results_dir	<path>:		Creates a new directory to store the tests' results into (default: running directory).

\e[4;37m\e[1;37mNotes\e[0m: 
(1) The efficiency tests are quite long, take that into account.
(2) Detailed regular tests' results are saved into \`\033[4;37mtest_transcript_<interface>.txt\e[0m\`.
(3) Detailed memory leak tests' results are saved into \`\033[4;37mmemory_transcript_<interface>.txt\e[0m\`.
(4) Detailed Efficiency tests' results are saved into \`\033[4;37mefficiency_transcript_<interface>.txt\e[0m\`.
(5) Only failed tests' results are saved into their transcript.
(6) Efficiency tests become invalid when running on tau-related server (e.g., nova).
(7) Python memory test is very slow (10 minutes) and doesn't run on Nova.

\e[4;37m\e[1;37mInstructions\e[0m:
(1) Avoid any build/dist/egg directories/files from the working directory. Could potentially lead to undefined behaviors of the test script.
(2) You shall run this shell script from within the directory that contains all of the files that you need to assign.
(3) You shall have the package \`\e[4;37mvalgrind\e[0m\` installed if you wish to run memory leak tests.
(4) You shall have the package \`\e[4;37mpython3.8-dbg\e[0m\` installed if you wish to run python's memory leak tests.
You can try to replace \`\e[4;37mpython3.8-dbg\e[0m\` with \`\e[4;37mpython3.x-dbg\e[0m\` or \`\e[4;37mpython3.xd\e[0m\` but make sure to change line 138 accordingly.
(5) You shall have the python packages \`\e[4;37mnumpy\e[0m\` and \`\e[4;37mscikit-learn\e[0m\` installed if you wish to run efficiency tests."
	exit
}



function youre_a_bozo() {
	echo -e "\e[0m
 _               ____      _  _____ ___ ___            ____   ___ ________  
| |        _    |  _ \    / \|_   _|_ _/ _ \     _    | __ ) / _ \__  / _ \ 
| |      _| |_  | |_) |  / _ \ | |  | | | | |  _| |_  |  _ \| | | |/ / | | |
| |___  |_   _| |  _ <  / ___ \| |  | | |_| | |_   _| | |_) | |_| / /| |_| |
|_____|   |_|   |_| \_\/_/   \_\_| |___\___/    |_|   |____/ \___/____\___/ 
                                                                            
          ____ ___  ____  _____           ____  _  _____ _     _     
   _     / ___/ _ \|  _ \| ____|    _    / ___|| |/ /_ _| |   | |    
 _| |_  | |  | | | | |_) |  _|    _| |_  \___ \| ' / | || |   | |    
|_   _| | |__| |_| |  __/| |___  |_   _|  ___) | . \ | || |___| |___ 
  |_|    \____\___/|_|   |_____|   |_|   |____/|_|\_\___|_____|_____|
                                                                     
 ___ ____ ____  _   _ _____ 
|_ _/ ___/ ___|| | | | ____|
 | |\___ \___ \| | | |  _|  
 | | ___) |__) | |_| | |___ 
|___|____/____/ \___/|_____|"
	exit
}



# blah0
if [[ $1 == "--help" ]]; then # help
	instructions
fi

# blah1
if [[ $# -le 3 ]]; then # args
	instructions
	youre_a_bozo
fi

# blah2
if [[ ! -d $1 ]]; then # testfiles
	echo -e "\e[1;31mThe following directory is a non-existing directory \`\033[4;31m${1}\e[0m\e[1;31m\`!\n"
	youre_a_bozo
else
	testers_path=$1
	jacobi=$(ls $testers_path | grep "jacobi" | cut -d "_" -f2 | cut -d "." -f1 | sort -n | tail -n1)
	spk=$(ls $testers_path | grep "spk" | cut -d "_" -f2 | cut -d "." -f1 | sort -n | tail -n1)
fi

# blah3
if [[ $2 != @(c|py|both) ]]; then # interface 
	youre_a_bozo
else
	interface=$2
fi

# blah4
if [[ $3 != @(yes|no) ]]; then # leaks
	youre_a_bozo
else
	regular=$3
fi

# blah5
if [[ $4 != @(yes|no) ]]; then # leaks
	youre_a_bozo
else
	leaks=$4
fi

# blah6
if [[ $5 != @(yes|no) ]]; then # efficiency
	youre_a_bozo
else
	efficiency=$5
fi

# blah7
if [[ $# -eq 6 ]]; then # results dir
	if [[ ! -d $6 ]]; then
		mkdir $6
	fi
	results_dir=$6
else
	results_dir="."
fi

# run
comprehensive_test
