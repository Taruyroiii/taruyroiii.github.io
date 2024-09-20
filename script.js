let options = {
    timeZone: 'Asia/Manila',
    year: 'numeric',
    month: 'numeric',
    day: 'numeric',
    hour: 'numeric',
    minute: 'numeric',
    second: 'numeric',
}

dateFormatter = new Intl.DateTimeFormat([], options);

// console.log(dateformatter.format(new Date()));
