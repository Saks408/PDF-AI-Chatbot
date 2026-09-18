const BACKEND_URL = "http://127.0.0.1:8000";

let currentChatId = null;
let pdfUploaded = false;


// ====================================
// DOM ELEMENTS
// ====================================

const pdfFile =
    document.getElementById("pdfFile");

const dropZone =
    document.getElementById("dropZone");

const selectedFile =
    document.getElementById("selectedFile");

const uploadStatus =
    document.getElementById("uploadStatus");

const questionInput =
    document.getElementById("question");

const messages =
    document.getElementById("messages");

const sendBtn =
    document.getElementById("sendBtn");

const uploadBtn =
    document.getElementById("uploadBtn");

const toast =
    document.getElementById("toast");


// ====================================
// START
// ====================================

document.addEventListener(
    "DOMContentLoaded",
    async () => {

        await loadDocuments();

        await loadChats();

    }
);


// ====================================
// LOAD ALL DOCUMENTS
// ====================================

async function loadDocuments() {

    try {

        const response =
            await fetch(
                `${BACKEND_URL}/documents`
            );


        if (!response.ok) {

            throw new Error(
                "Could not load documents"
            );

        }


        const data =
            await response.json();


        pdfUploaded =
            data.documents &&
            data.documents.length > 0;


        renderDocuments(
            data.documents || []
        );


    }

    catch (error) {

        console.error(
            "Document loading error:",
            error
        );

    }

}

// ====================================
// RENDER DOCUMENT LIST
// ====================================

function renderDocuments(docs) {

    const container =
        document.getElementById("documentList");

    if (!container) {

        console.warn(
            "documentList element not found"
        );

        return;
    }

    container.innerHTML = "";

    if (!docs || docs.length === 0) {

        container.innerHTML = `
            <div class="empty-documents">
                No PDF uploaded
            </div>
        `;

        if (uploadStatus) {
            uploadStatus.textContent =
                "No PDF uploaded.";
        }

        pdfUploaded = false;

        return;
    }

    pdfUploaded = true;

    docs.forEach(doc => {

        const item =
            document.createElement("div");

        item.className =
            "document-item";


        // ====================================
        // LEFT SIDE
        // ====================================

        const info =
            document.createElement("div");

        info.className =
            "document-info";


        // ====================================
        // PDF ICON
        // ====================================

        const icon =
            document.createElement("span");

        icon.className =
            "document-icon";

        icon.textContent =
            "📄";


        // ====================================
        // DETAILS
        // ====================================

        const details =
            document.createElement("div");

        details.className =
            "document-details";


        // ====================================
        // FILE NAME
        // ====================================

        const name =
            document.createElement("div");

        name.className =
            "document-name";

        name.textContent =
            doc.filename;


        // ====================================
        // META
        // ====================================

        const meta =
            document.createElement("div");

        meta.className =
            "document-meta";

        meta.textContent =
            `${doc.pages} pages • ${doc.chunks} chunks`;


        // ====================================
        // BUILD DETAILS
        // ====================================

        details.appendChild(name);

        details.appendChild(meta);


        info.appendChild(icon);

        info.appendChild(details);


        // ====================================
        // REMOVE BUTTON
        // ====================================

        const removeBtn =
            document.createElement("button");

        removeBtn.className =
            "remove-document";

        removeBtn.textContent =
            "🗑";

        removeBtn.title =
            "Remove PDF";


        removeBtn.onclick =
            event => {

                event.stopPropagation();

                removeDocument(
                    doc.filename
                );

            };


        // ====================================
        // FINAL ITEM
        // ====================================

        item.appendChild(info);

        item.appendChild(removeBtn);

        container.appendChild(item);

    });


    // ====================================
    // STATUS
    // ====================================

    if (uploadStatus) {

        uploadStatus.textContent =
            `${docs.length} PDF${
                docs.length > 1
                    ? "s"
                    : ""
            } ready for questions.`;

    }

}
    uploadStatus.textContent =
        `${documents.length} PDF${documents.length > 1 ? "s" : ""} ready for questions.`;




// ====================================
// UPLOAD PDF
// ====================================

if (uploadBtn) {

    uploadBtn.addEventListener(
        "click",
        uploadPDF
    );

}


async function uploadPDF() {

    const files =
        pdfFile.files;


    if (!files || files.length === 0) {

        showToast(
            "Please select PDF file(s)."
        );

        return;

    }


    uploadBtn.disabled = true;


    let successCount = 0;


    try {

        for (
            let i = 0;
            i < files.length;
            i++
        ) {

            const file =
                files[i];


            if (
                !file.name
                    .toLowerCase()
                    .endsWith(".pdf")
            ) {

                showToast(
                    `${file.name} is not a PDF.`
                );

                continue;

            }


            const formData =
                new FormData();


            formData.append(
                "file",
                file
            );


            uploadStatus.textContent =
                `Uploading ${file.name}...`;


            const response =
                await fetch(
                    `${BACKEND_URL}/upload`,
                    {
                        method: "POST",
                        body: formData
                    }
                );


            const data =
                await response.json();


            if (!response.ok) {

                throw new Error(
                    data.detail ||
                    `Upload failed: ${file.name}`
                );

            }


            successCount++;

        }


        await loadDocuments();


        pdfFile.value = "";


        if (selectedFile) {

            selectedFile.textContent =
                "";

        }


        showToast(
            `${successCount} PDF uploaded successfully.`
        );

    }

    catch (error) {

        console.error(
            "Upload error:",
            error
        );


        showToast(
            error.message ||
            "Could not upload PDF."
        );

    }

    finally {

        uploadBtn.disabled = false;

    }

}


// ====================================
// REMOVE PDF
// ====================================

async function removeDocument(
    filename
) {

    const confirmed =
        confirm(
            `Are you sure you want to remove "${filename}"?`
        );


    if (!confirmed) return;


    try {

        const response =
            await fetch(

                `${BACKEND_URL}/documents/${encodeURIComponent(filename)}`,

                {
                    method: "DELETE"
                }

            );


        const data =
            await response.json();


        if (!response.ok) {

            throw new Error(
                data.detail ||
                "Could not remove PDF."
            );

        }


        await loadDocuments();


        showToast(
            `${filename} removed.`
        );


    }

    catch (error) {

        console.error(
            "Remove PDF error:",
            error
        );


        showToast(
            error.message ||
            "Could not remove PDF."
        );

    }

}


// ====================================
// LOAD ALL CHATS
// ====================================

async function loadChats() {

    try {

        const response =
            await fetch(
                `${BACKEND_URL}/chats`
            );


        const data =
            await response.json();


        renderChatList(
            data.chats
        );


        if (
            data.chats &&
            data.chats.length > 0
        ) {

            const savedChat =
                localStorage.getItem(
                    "currentChatId"
                );


            const exists =
                data.chats.some(
                    chat =>
                        chat.id === savedChat
                );


            await selectChat(

                exists
                    ? savedChat
                    : data.chats[0].id

            );

        }

        else {

            await createChat();

        }

    }

    catch (error) {

        console.error(error);

        showToast(
            "Could not load chats."
        );

    }

}


// ====================================
// RENDER CHAT LIST
// ====================================

function renderChatList(chats) {

    const chatList =
        document.getElementById(
            "chatList"
        );


    if (!chatList) return;


    if (!chats.length) {

        chatList.innerHTML =
            `
            <div class="empty-chats">
                No saved chats
            </div>
            `;

        return;

    }


    chatList.innerHTML = "";


    chats.forEach(
        chat => {

            const item =
                document.createElement(
                    "div"
                );


            item.className =
                "chat-item " +
                (
                    chat.id === currentChatId
                        ? "active"
                        : ""
                );


            const main =
                document.createElement(
                    "div"
                );


            main.className =
                "chat-main";


            main.onclick =
                () =>
                    selectChat(
                        chat.id
                    );


            const name =
                document.createElement(
                    "span"
                );


            name.className =
                "chat-name";


            name.textContent =
                chat.title ||
                "New Chat";


            const date =
                document.createElement(
                    "span"
                );


            date.className =
                "chat-date";


            date.textContent =
                new Date(
                    chat.updated_at
                ).toLocaleString(
                    [],
                    {
                        day: "2-digit",
                        month: "short",
                        hour: "2-digit",
                        minute: "2-digit"
                    }
                );


            main.appendChild(name);

            main.appendChild(date);


            const deleteBtn =
                document.createElement(
                    "button"
                );


            deleteBtn.className =
                "delete-chat";


            deleteBtn.innerHTML =
                "🗑";


            deleteBtn.title =
                "Delete chat";


            deleteBtn.onclick =
                event => {

                    event.stopPropagation();

                    deleteChat(
                        chat.id
                    );

                };


            item.appendChild(main);

            item.appendChild(
                deleteBtn
            );


            chatList.appendChild(
                item
            );

        }
    );

}


// ====================================
// CREATE NEW CHAT
// ====================================

async function createChat() {

    try {

        const response =
            await fetch(

                `${BACKEND_URL}/chats`,

                {
                    method: "POST",

                    headers: {
                        "Content-Type":
                            "application/json"
                    },

                    body: JSON.stringify({
                        title: "New Chat"
                    })
                }

            );


        const chat =
            await response.json();


        currentChatId =
            chat.id;


        localStorage.setItem(
            "currentChatId",
            currentChatId
        );


        const chatTitle =
            document.getElementById(
                "chatTitle"
            );


        if (chatTitle) {

            chatTitle.textContent =
                "New Chat";

        }


        renderMessages(
            chat
        );


        await loadChats();

    }

    catch (error) {

        console.error(error);

        showToast(
            "Could not create chat."
        );

    }

}


// ====================================
// NEW CHAT
// ====================================

async function newChat() {

    await createChat();


    questionInput.value = "";

    autoResize();

    questionInput.focus();

}


// ====================================
// SELECT CHAT
// ====================================

async function selectChat(
    chatId
) {

    try {

        const response =
            await fetch(
                `${BACKEND_URL}/chats/${chatId}`
            );


        if (!response.ok) {

            throw new Error(
                "Chat not found"
            );

        }


        const chat =
            await response.json();


        currentChatId =
            chat.id;


        localStorage.setItem(
            "currentChatId",
            currentChatId
        );


        const chatTitle =
            document.getElementById(
                "chatTitle"
            );


        if (chatTitle) {

            chatTitle.textContent =
                chat.title ||
                "New Chat";

        }


        renderMessages(
            chat
        );


        const listResponse =
            await fetch(
                `${BACKEND_URL}/chats`
            );


        const listData =
            await listResponse.json();


        renderChatList(
            listData.chats
        );

    }

    catch (error) {

        console.error(error);

        showToast(
            "Could not open chat."
        );

    }

}


// ====================================
// DELETE CHAT
// ====================================

async function deleteChat(
    chatId
) {

    const confirmed =
        confirm(
            "Are you sure you want to delete this chat?"
        );


    if (!confirmed) return;


    try {

        const response =
            await fetch(

                `${BACKEND_URL}/chats/${chatId}`,

                {
                    method: "DELETE"
                }

            );


        if (!response.ok) {

            throw new Error(
                "Delete failed"
            );

        }


        if (
            chatId ===
            currentChatId
        ) {

            currentChatId =
                null;


            localStorage.removeItem(
                "currentChatId"
            );

        }


        await loadChats();


        showToast(
            "Chat deleted."
        );

    }

    catch (error) {

        console.error(error);

        showToast(
            "Could not delete chat."
        );

    }

}


// ====================================
// CLEAR CHAT
// ====================================

async function clearCurrentChat() {

    if (!currentChatId) return;


    const confirmed =
        confirm(
            "Clear all messages from this chat?"
        );


    if (!confirmed) return;


    try {

        const response =
            await fetch(

                `${BACKEND_URL}/chats/${currentChatId}/clear`,

                {
                    method: "POST"
                }

            );


        const chat =
            await response.json();


        renderMessages(
            chat
        );


        const chatTitle =
            document.getElementById(
                "chatTitle"
            );


        if (chatTitle) {

            chatTitle.textContent =
                "New Chat";

        }


        await loadChats();


        showToast(
            "Chat cleared."
        );

    }

    catch (error) {

        console.error(error);

        showToast(
            "Could not clear chat."
        );

    }

}


// ====================================
// ASK QUESTION
// ====================================

async function askQuestion() {

    const question =
        questionInput.value.trim();


    if (!question) {

        showToast(
            "Please enter a question."
        );

        return;

    }


    if (!pdfUploaded) {

        await loadDocuments();

    }


    if (!pdfUploaded) {

        showToast(
            "Upload at least one PDF first."
        );

        return;

    }


    if (!currentChatId) {

        await createChat();

    }


    addMessage(
        "user",
        question
    );


    questionInput.value = "";

    autoResize();

    sendBtn.disabled = true;


    const loading =
        addMessage(
            "bot",
            "",
            true
        );


    try {

        const response =
            await fetch(

                `${BACKEND_URL}/chats/${currentChatId}/ask`,

                {

                    method: "POST",

                    headers: {
                        "Content-Type":
                            "application/json"
                    },

                    body: JSON.stringify({
                        question:
                            question
                    })

                }

            );


        const data =
            await response.json();


        if (!response.ok) {

            throw new Error(
                data.detail ||
                "Question failed"
            );

        }


        loading.remove();


        addMessage(
            "bot",
            data.answer
        );


        const chatTitle =
            document.getElementById(
                "chatTitle"
            );


        if (chatTitle) {

            chatTitle.textContent =
                data.title;

        }


        const chatsResponse =
            await fetch(
                `${BACKEND_URL}/chats`
            );


        const chatsData =
            await chatsResponse.json();


        renderChatList(
            chatsData.chats
        );

    }

    catch (error) {

        loading.remove();


        addMessage(
            "bot",
            `Sorry, I couldn't process that question.\n\n${error.message}`
        );

    }

    finally {

        sendBtn.disabled = false;

        questionInput.focus();

    }

}


// ====================================
// RENDER MESSAGES
// ====================================

function renderMessages(
    chat
) {

    messages.innerHTML = "";


    if (
        !chat.messages ||
        chat.messages.length === 0
    ) {

        addMessage(
            "bot",
            "Hi! I'm your PDF assistant. Upload one or more PDFs above, then ask me anything about their content."
        );

        return;

    }


    chat.messages.forEach(
        message => {

            addMessage(

                message.role === "user"
                    ? "user"
                    : "bot",

                message.content,

                false,

                message.created_at

            );

        }
    );

}


// ====================================
// ADD MESSAGE
// ====================================

function addMessage(
    type,
    text,
    isLoading = false,
    time = null
) {

    const isUser =
        type === "user";


    const message =
        document.createElement(
            "div"
        );


    message.className =
        `message ${
            isUser
                ? "user-message"
                : "bot-message"
        }`;


    const avatar =
        document.createElement(
            "div"
        );


    avatar.className =
        `avatar ${
            isUser
                ? "user-avatar"
                : "bot-avatar"
        }`;


    avatar.textContent =
        isUser
            ? "👤"
            : "🤖";


    const content =
        document.createElement(
            "div"
        );


    content.className =
        "message-content";


    const bubble =
        document.createElement(
            "div"
        );


    bubble.className =
        "message-bubble";


    if (isLoading) {

        bubble.innerHTML =
            `
            <div class="typing">
                <span></span>
                <span></span>
                <span></span>
            </div>
            `;

    }

    else {

        bubble.textContent =
            text;

    }


    const timestamp =
        document.createElement(
            "span"
        );


    timestamp.className =
        "message-time";


    timestamp.textContent =
        getTime(time);


    content.appendChild(
        bubble
    );

    content.appendChild(
        timestamp
    );


    message.appendChild(
        avatar
    );

    message.appendChild(
        content
    );


    messages.appendChild(
        message
    );


    messages.scrollTop =
        messages.scrollHeight;


    return message;

}


// ====================================
// TIME
// ====================================

function getTime(
    time = null
) {

    const date =
        time
            ? new Date(time)
            : new Date();


    return date.toLocaleTimeString(
        [],
        {
            hour: "2-digit",
            minute: "2-digit"
        }
    );

}


// ====================================
// ENTER TO SEND
// ====================================

function handleEnter(
    event
) {

    if (
        event.key === "Enter" &&
        !event.shiftKey
    ) {

        event.preventDefault();

        askQuestion();

    }

}


// ====================================
// TEXTAREA
// ====================================

function autoResize() {

    questionInput.style.height =
        "auto";


    questionInput.style.height =
        Math.min(
            questionInput.scrollHeight,
            90
        ) + "px";

}


questionInput.addEventListener(
    "input",
    autoResize
);


// ====================================
// SUGGESTIONS
// ====================================

function useSuggestion(
    text
) {

    questionInput.value =
        text;

    autoResize();

    questionInput.focus();

}


// ====================================
// TOAST
// ====================================

function showToast(
    message
) {

    if (!toast) return;


    toast.textContent =
        message;


    toast.classList.add(
        "show"
    );


    setTimeout(
        () => {

            toast.classList.remove(
                "show"
            );

        },

        2500
    );

}