if [ $1 = "32" ];
    then
        sed -n '\|read_|,$p' ../log | head -n -1 | awk -F' ' '{print $5}' > $2;
    else
         sed -n '\|read_|,$p' ../log | head -n -1 | awk -F' ' '{print $5}' > "$2.fwd";
         sed -n '\|read_|,$p' ../log | head -n -1 | awk -F' ' '{print $6}' > "$2.rev";
fi